"""Command line interface for :mod:`mcp_plex.server`."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from fastapi import FastAPI
from fastmcp.server.middleware.logging import StructuredLoggingMiddleware
import uvicorn

from . import PlexServer, server, settings


plex_server: PlexServer = server


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


DUAL_TRANSPORT = "sse+streamable-http"


def _resolve_log_level(cli_value: str | None) -> str:
    """Return the desired log level name based on CLI or environment input."""

    env_value = os.getenv("LOG_LEVEL")
    if cli_value:
        return cli_value
    if env_value:
        return env_value.lower()
    return "info"


def _normalize_transport(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"both", DUAL_TRANSPORT, "streamable-http+sse"}:
        return DUAL_TRANSPORT
    if "," in normalized:
        tokens = {token.strip() for token in normalized.split(",") if token.strip()}
        if tokens == {"sse", "streamable-http"}:
            return DUAL_TRANSPORT
    return normalized


def _normalize_mount(mount: str) -> str:
    normalized = mount.strip()
    if not normalized:
        raise ValueError("mount must not be empty")
    if not normalized.startswith("/"):
        normalized = f"/{normalized}"
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")
    return normalized


def _run_dual_http_transports(
    *,
    server_instance: PlexServer,
    host: str,
    port: int,
    sse_mount: str,
    streamable_http_mount: str,
    log_level: str,
) -> None:
    app = FastAPI()
    app.mount(sse_mount, server_instance.http_app(path="/", transport="sse"))
    app.mount(
        streamable_http_mount,
        server_instance.http_app(path="/", transport="streamable-http"),
    )

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=0,
        lifespan="on",
        ws="websockets-sansio",
        log_level=log_level,
    )
    uvicorn.Server(config).run()


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running the MCP server."""

    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument("--bind", help="Host address to bind to")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http", "both"],
        default="stdio",
        help="Transport protocol to use",
    )
    parser.add_argument("--mount", help="Mount path for HTTP transports")
    parser.add_argument("--sse-mount", help="Mount path for SSE when using --transport both")
    parser.add_argument(
        "--streamable-http-mount",
        help="Mount path for streamable HTTP when using --transport both",
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

    transport = _normalize_transport(env_transport or args.transport)
    valid_transports = {"stdio", "sse", "streamable-http", DUAL_TRANSPORT}
    if transport not in valid_transports:
        parser.error(
            "transport must be one of stdio, sse, streamable-http, or both (via --transport or MCP_TRANSPORT)"
        )

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

    if transport != "stdio":
        if host is None or port is None:
            parser.error(
                "--bind/--port or MCP_HOST/MCP_PORT are required when transport is not stdio"
            )
    if transport == "stdio" and mount:
        parser.error("--mount or MCP_MOUNT is not allowed when transport is stdio")

    env_sse_mount = os.getenv("MCP_SSE_MOUNT")
    env_streamable_mount = os.getenv("MCP_STREAMABLE_HTTP_MOUNT")

    sse_mount = env_sse_mount or args.sse_mount
    streamable_mount = env_streamable_mount or args.streamable_http_mount or mount

    run_config = RunConfig()
    if transport != "stdio" and transport != DUAL_TRANSPORT:
        if host is not None:
            run_config.host = host
        if port is not None:
            run_config.port = port
        if mount:
            run_config.path = mount

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
            include_payloads=True,
            log_level=log_level
        )
    )
    if transport == DUAL_TRANSPORT:
        if not host or port is None:
            parser.error(
                "--bind/--port or MCP_HOST/MCP_PORT are required when transport is both"
            )
        if not sse_mount or not streamable_mount:
            parser.error(
                "--sse-mount/--streamable-http-mount or MCP_SSE_MOUNT/MCP_STREAMABLE_HTTP_MOUNT are required when transport is both"
            )
        try:
            normalized_sse_mount = _normalize_mount(sse_mount)
            normalized_streamable_mount = _normalize_mount(streamable_mount)
        except ValueError:
            parser.error("mount values must not be empty")
        if normalized_sse_mount == normalized_streamable_mount:
            parser.error(
                "SSE and streamable HTTP mounts must be different when transport is both"
            )
        _run_dual_http_transports(
            server_instance=plex_server,
            host=host,
            port=port,
            sse_mount=normalized_sse_mount,
            streamable_http_mount=normalized_streamable_mount,
            log_level=log_level_name,
        )
        return
    plex_server.run(transport=transport, **run_config.to_kwargs())


__all__ = [
    "DUAL_TRANSPORT",
    "RunConfig",
    "main",
    "server",
    "PlexServer",
    "plex_server",
    "settings",
]
