# AGENTS

## Architecture
- `mcp_plex.server` exposes retrieval and search tools via FastMCP backed by Qdrant.
- Hybrid search uses Qdrant's built-in `FusionQuery` with reciprocal rank fusion to combine dense and sparse results before optional cross-encoder reranking.
- The Qdrant client is initialized inside `PlexServer` to centralize state and simplify testing.
- The cross-encoder reranker is loaded lazily via a `PlexServer` property so models are only downloaded when reranking is enabled and available.
- Media payload and artwork caching are centralized in a `MediaCache` attached to `PlexServer` for consistent cache management across endpoints.

