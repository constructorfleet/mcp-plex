# ADR 0003: Allow SSE and Streamable HTTP to Coexist in One Process

## Status
Accepted

## Context
The server CLI previously treated transport selection as mutually exclusive, so operators had to choose either SSE or streamable HTTP for a given process. That made deployment awkward when a single container or host should serve both transports on the same IP and port, but at different URL paths for clients with different transport expectations.

## Decision
The CLI now accepts repeatable `--transport` arguments and a comma-delimited `MCP_TRANSPORT` value so a single process can expose multiple HTTP transports at once. When both SSE and streamable HTTP are selected, the CLI builds one parent ASGI app and mounts separate FastMCP HTTP apps beneath transport-specific paths. SSE defaults to `/sse` and streamable HTTP defaults to `/mcp`, and both paths can be overridden independently through CLI flags or environment variables.

`stdio` remains exclusive. It cannot be combined with HTTP transports in the same process.

## Consequences
- Operators can run SSE and streamable HTTP on one host and port without starting two processes.
- Transport-specific mount points are explicit, which avoids accidental route collisions.
- The CLI surface is slightly larger, but the default single-transport behavior stays unchanged.
- Documentation and tests must cover the shared-process case and the transport-specific mount overrides.

## Implementation Notes
- The CLI composes the shared-process ASGI app in `mcp_plex/server/cli.py` using Starlette mounts and FastMCP HTTP child apps.
- The default `--mount`/`MCP_MOUNT` behavior remains available for a single HTTP transport.
- If both HTTP transports are enabled, the CLI requires distinct mount paths and rejects the legacy generic mount override.
