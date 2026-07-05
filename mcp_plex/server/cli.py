"""Command line interface for :mod:`mcp_plex.server`."""

from __future__ import annotations

import argparse
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Literal

import uvicorn
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.types import ASGIApp, Receive, Scope, Send

from . import PlexServer, server, settings


plex_server: PlexServer = server
HTTP_TRANSPORTS = ("sse", "streamable-http")
VALID_TRANSPORTS = ("stdio", *HTTP_TRANSPORTS)
DEFAULT_HTTP_MOUNTS = {
    "sse": "/sse",
    "streamable-http": "/mcp",
}


@dataclass(frozen=True)
class HttpTransportConfig:
    """Resolved HTTP transport configuration for a shared-process server."""

    transport: Literal["sse", "streamable-http"]
    path: str


@dataclass
class RunConfig:
    """Runtime configuration for FastMCP transport servers."""

    host: str | None = None
    port: int | None = None
    path: str | None = None

    def to_kwargs(self) -> dict[str, object]:
        """Return keyword arguments compatible with ``FastMCP.run``."""

        kwargs: dict[str, object] = {}
        if self.host is not None:
            kwargs["host"] = self.host
        if self.port is not None:
            kwargs["port"] = self.port
        if self.path:
            kwargs["path"] = self.path
        return kwargs


def _normalize_mount_path(value: str | None) -> str | None:
    """Return a canonical HTTP mount path or ``None`` when unset."""

    if value is None:
        return None
    mount = value.strip()
    if not mount:
        return None
    if not mount.startswith("/"):
        mount = f"/{mount}"
    if mount != "/":
        mount = mount.rstrip("/")
    return mount or "/"


def _parse_transport_list(value: str | None) -> list[str]:
    """Split a comma-delimited transport list into normalized names."""

    if value is None:
        return []
    return [transport for transport in (item.strip() for item in value.split(",")) if transport]


def _resolve_transports(
    cli_values: list[str] | None,
    env_value: str | None,
    parser: argparse.ArgumentParser,
) -> list[str]:
    """Return the transport selection with environment overrides applied."""

    if env_value is not None:
        transports = _parse_transport_list(env_value)
        if not transports:
            parser.error("MCP_TRANSPORT must list at least one transport")
    elif cli_values:
        transports = cli_values
    else:
        transports = ["stdio"]

    invalid = sorted(set(transports) - set(VALID_TRANSPORTS))
    if invalid:
        parser.error(
            "transport must be one of stdio, sse, or streamable-http (via --transport or MCP_TRANSPORT)"
        )

    if "stdio" in transports and len(transports) > 1:
        parser.error("stdio cannot be combined with SSE or streamable-http")

    deduped: list[str] = []
    seen: set[str] = set()
    for transport in transports:
        if transport not in seen:
            deduped.append(transport)
            seen.add(transport)

    return deduped


def _resolve_http_mount(
    transport: Literal["sse", "streamable-http"],
    *,
    multiple_http_transports: bool,
    generic_mount: str | None,
    transport_mount: str | None,
    parser: argparse.ArgumentParser,
) -> str:
    """Resolve a transport-specific HTTP mount path."""

    if multiple_http_transports:
        if generic_mount is not None:
            parser.error(
                "--mount or MCP_MOUNT cannot be used when running both SSE and streamable-http; use --sse-mount and --streamable-http-mount"
            )
        mount = transport_mount or DEFAULT_HTTP_MOUNTS[transport]
    else:
        mount = transport_mount or generic_mount or DEFAULT_HTTP_MOUNTS[transport]
    normalized = _normalize_mount_path(mount)
    if normalized is None:
        parser.error(f"{transport} mount path could not be resolved")
    return normalized


class _TrailingSlashMiddleware:
    """Append a trailing slash to requests that exactly match a mount path.

    Starlette's ``Mount`` compiles a regex of the form ``^/path/(?P<path>.*)$``,
    so it requires the trailing slash even though the captured suffix can be
    empty. A plain ``GET /sse`` therefore falls through every ``Mount`` and
    Starlette's built-in ``redirect_slashes`` emits a 307 → ``/sse/``. Some MCP
    clients (e.g. Home Assistant) do not follow 307 redirects, which prevents
    them from connecting.

    This middleware transparently rewrites an exact-path hit (``/sse``) to
    ``/sse/`` *before* the router sees it, so the ``Mount`` can match without
    ever sending a redirect to the client.
    """

    def __init__(self, app: ASGIApp, *, paths: frozenset[str]) -> None:
        self.app = app
        self.paths = paths

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path") in self.paths:
            scope = {**scope, "path": scope["path"] + "/"}
        await self.app(scope, receive, send)


def _build_shared_http_app(configs: list[HttpTransportConfig]) -> Starlette:
    """Build a single ASGI app that exposes multiple FastMCP HTTP transports."""

    child_apps = []
    for config in configs:
        child_app = plex_server.http_app(path="/", transport=config.transport)
        child_apps.append((config.path, child_app))

    @asynccontextmanager
    async def lifespan(_app: Starlette):
        async with AsyncExitStack() as stack:
            for _, child_app in child_apps:
                await stack.enter_async_context(child_app.router.lifespan_context(child_app))
            yield
    mount_paths: frozenset[str] = frozenset(
        config.path for config in configs if config.path != "/"
    )
    app = Starlette(
        lifespan=lifespan,
        middleware=[Middleware(_TrailingSlashMiddleware, paths=mount_paths)],
    )
    if all(config.path != "/" for config in configs):
        for route in plex_server._get_additional_http_routes():
            methods = [m for m in (route.methods or {"GET"}) if m != "HEAD"]
            app.add_route(route.path, route.endpoint, methods=methods)
    for mount_path, child_app in sorted(
        child_apps, key=lambda entry: (entry[0] == "/", -len(entry[0]))
    ):
        app.mount(mount_path, child_app)
    return app


def _resolve_log_level(cli_value: str | None) -> str:
    """Return the desired log level name based on CLI or environment input."""

    env_value = os.getenv("LOG_LEVEL")
    if cli_value:
        return cli_value
    if env_value:
        return env_value.lower()
    return "info"


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running the MCP server."""

    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument("--bind", help="Host address to bind to")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument(
        "--transport",
        action="append",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport protocol to use",
    )
    parser.add_argument("--mount", help="Mount path for HTTP transports")
    parser.add_argument("--sse-mount", help="Mount path for SSE transports")
    parser.add_argument(
        "--streamable-http-mount",
        help="Mount path for streamable HTTP transports",
    )
    parser.add_argument(
        "--dense-model",
        default=settings.dense_model,
        help="Dense embedding model name (env: DENSE_MODEL)",
    )
    parser.add_argument(
        "--sparse-model",
        default=settings.sparse_model,
        help="Sparse embedding model name (env: SPARSE_MODEL)",
    )
    parser.add_argument(
        "--reranker-model",
        default=settings.reranker_model,
        help="Cross-encoder reranker model name (env: RERANKER_MODEL)",
    )
    parser.add_argument(
        "--recommend-user",
        default=settings.recommend_user,
        help="Plex username whose watch history should be excluded (env: PLEX_RECOMMEND_USER)",
    )
    parser.add_argument(
        "--recommend-history-limit",
        type=int,
        default=settings.recommend_history_limit,
        help=(
            "Maximum number of history entries to exclude per user (env: PLEX_RECOMMEND_HISTORY_LIMIT)"
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str.lower,
        choices=["critical", "error", "warning", "info", "debug", "notset"],
        help="Logging verbosity (env: LOG_LEVEL)",
    )
    args = parser.parse_args(argv)

    env_transport = os.getenv("MCP_TRANSPORT")
    env_host = (
        os.getenv("MCP_HOST")
        if os.getenv("MCP_HOST") is not None
        else os.getenv("MCP_BIND")
    )
    env_port = os.getenv("MCP_PORT")
    env_mount = os.getenv("MCP_MOUNT")
    env_sse_mount = os.getenv("MCP_SSE_MOUNT")
    env_streamable_http_mount = os.getenv("MCP_STREAMABLE_HTTP_MOUNT")

    transports = _resolve_transports(args.transport, env_transport, parser)

    host = env_host or args.bind
    port: int | None
    if env_port is not None:
        try:
            port = int(env_port)
        except ValueError:
            parser.error("MCP_PORT must be an integer")
    else:
        port = args.port

    mount = env_mount or args.mount
    sse_mount = env_sse_mount or args.sse_mount
    streamable_http_mount = env_streamable_http_mount or args.streamable_http_mount
    http_transports = [transport for transport in transports if transport != "stdio"]

    if http_transports:
        if host is None or port is None:
            parser.error(
                "--bind/--port or MCP_HOST/MCP_PORT are required when transport is not stdio"
            )
    if transports == ["stdio"] and (mount or sse_mount or streamable_http_mount):
        parser.error(
            "--mount/--sse-mount/--streamable-http-mount (and MCP_MOUNT/MCP_SSE_MOUNT/MCP_STREAMABLE_HTTP_MOUNT) are not allowed when transport is stdio"
        )

    settings.dense_model = args.dense_model
    settings.sparse_model = args.sparse_model
    settings.reranker_model = args.reranker_model
    settings.recommend_user = args.recommend_user
    settings.recommend_history_limit = max(0, args.recommend_history_limit)

    log_level_name = _resolve_log_level(args.log_level)
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    logging.basicConfig(level=log_level)
    plex_server.add_middleware(
        StructuredLoggingMiddleware(
            include_payloads=True, log_level=log_level
        )
    )

    if transports == ["stdio"]:
        plex_server.run(transport="stdio")
        return

    if len(http_transports) == 1:
        transport = http_transports[0]
        run_config = RunConfig()
        if host is not None:
            run_config.host = host
        if port is not None:
            run_config.port = port
        resolved_mount = _resolve_http_mount(
            transport,  # type: ignore[arg-type]
            multiple_http_transports=False,
            generic_mount=mount,
            transport_mount=sse_mount if transport == "sse" else streamable_http_mount,
            parser=parser,
        )
        if resolved_mount != DEFAULT_HTTP_MOUNTS[transport]:
            run_config.path = resolved_mount
        plex_server.run(transport=transport, **run_config.to_kwargs())
        return

    multi_http_configs = [
        HttpTransportConfig(
            transport=transport,  # type: ignore[arg-type]
            path=_resolve_http_mount(
                transport,  # type: ignore[arg-type]
                multiple_http_transports=True,
                generic_mount=mount,
                transport_mount=(
                    sse_mount if transport == "sse" else streamable_http_mount
                ),
                parser=parser,
            ),
        )
        for transport in http_transports
    ]
    resolved_paths = [config.path for config in multi_http_configs]
    if len(set(resolved_paths)) != len(resolved_paths):
        parser.error("SSE and streamable-http must use different mount paths")

    shared_app = _build_shared_http_app(multi_http_configs)
    uvicorn.run(shared_app, host=host, port=port, log_level=log_level_name)


__all__ = ["RunConfig", "main", "server", "PlexServer", "plex_server", "settings"]
