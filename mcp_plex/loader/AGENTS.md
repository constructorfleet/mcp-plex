# AGENTS

## Architecture
- `mcp_plex.loader` ingests Plex, TMDb, and IMDb metadata, generates dense and sparse embeddings with Qdrant, and stores the items in a Qdrant collection.
- Embedding generation uses Qdrant's document API so no local embedding models need to be bundled.
- Actor names are stored as a top-level payload field and indexed in Qdrant to support actor and year-based filtering.
- Dense and sparse embedding model names are configurable via the `DENSE_MODEL` and `SPARSE_MODEL` environment variables or the corresponding CLI options.
- Plex episodes with year-based seasons are mapped to the correct TMDb season numbers by matching season names or air-date years.
- Plex metadata is fetched in batches using `fetchItems` to minimize repeated network calls when loading libraries.
- IMDb metadata is fetched via the `titles:batchGet` endpoint, which accepts at most five IDs per request, so lookups are split into batches of five.
- Qdrant upserts are batched and network errors are logged so large loads can continue even when individual batches fail.
- Qdrant model metadata is tracked locally to avoid relying on private client helpers.
- Qdrant collection setup happens before media ingestion, and the loader streams asynchronous upsert tasks once the configurable buffer fills so fetching can continue while points are written.

