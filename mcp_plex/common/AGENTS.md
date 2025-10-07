# AGENTS

## Architecture
- `mcp_plex.common` provides shared cache helpers, data models, and utility types consumed by both the loader and the server packages.
- Keep shared logic decoupled from CLI wiring so it can be imported safely by tests and other packages.
- Update this module when adding reusable functionality to avoid duplicating code between the loader and server implementations.
- Use the `JSONValue` and related aliases in `types.py` when exchanging cached payloads or structured JSON-like data. Media caches and downstream consumers expect payload dictionaries to resolve to `dict[str, JSONValue]` without falling back to ``Any``.

