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
- The staged loader rewrite lives under `mcp_plex/loader/pipeline/`.  The concrete classes that must be wired together are:
  - `IngestionStage` (`ingestion.py`)
  - `EnrichmentStage` (`enrichment.py`)
  - `PersistenceStage` (`persistence.py`)
  - `LoaderOrchestrator` (`orchestrator.py`)
- `mcp_plex/loader/pipeline/channels.py` defines the queue type aliases and sentinel tokens (`INGEST_DONE`, `PERSIST_DONE`) shared by the stages.

## Loader CLI expectations
- `mcp_plex/loader/__init__.py` exposes helpers for wiring the staged loader pipeline. New work should instantiate the stages directly and coordinate them with `LoaderOrchestrator`.
- When constructing stages from the CLI:
  - `IngestionStage` must receive the Plex server (or `None` for sample mode), the list of sample items, the Plex chunk size for both movies and episodes, the enrichment batch size for sample batches, the ingest queue instance, and the `INGEST_DONE` sentinel.
  - `EnrichmentStage` requires a factory that returns an `httpx.AsyncClient` (or context manager), the TMDb API key (empty string when unused in sample mode), the ingest queue, the persistence queue, the shared `IMDbRetryQueue`, the enrichment batch size for movies and episodes, and the IMDb configuration derived from CLI flags.
  - `PersistenceStage` expects the `AsyncQdrantClient`, collection name, dense/sparse vector names, the persistence queue, the Qdrant retry queue, the semaphore limiting concurrent upserts, the upsert buffer size, and callables for performing the upsert as well as recording progress.
  - `LoaderOrchestrator` must be initialised with the three stage instances, the ingest queue, the persistence queue, and the number of persistence workers (the CLI's `max_concurrent_upserts`).
- Convert `AggregatedItem` batches into Qdrant `PointStruct` objects with `build_point` before handing them to the persistence stage's `enqueue_points` helper.
- Prefer explicit keyword arguments when threading CLI options into stage constructors so the mapping is obvious to future readers.
