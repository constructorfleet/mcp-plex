"""Command line interface for :mod:`mcp_plex.server`."""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

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
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use",
    )
    parser.add_argument("--mount", help="Mount path for HTTP transports")
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

    transport = env_transport or args.transport
    valid_transports = {"stdio", "sse", "streamable-http"}
    if transport not in valid_transports:
        parser.error(
            "transport must be one of stdio, sse, or streamable-http (via --transport or MCP_TRANSPORT)"
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

    run_config = RunConfig()
    if transport != "stdio":
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

    log_level_name = _resolve_log_level(args.log_level)
    logging.basicConfig(level=getattr(logging, log_level_name.upper(), logging.INFO))

    plex_server.run(transport=transport, **run_config.to_kwargs())


__all__ = ["RunConfig", "main", "server", "PlexServer", "plex_server", "settings"]
