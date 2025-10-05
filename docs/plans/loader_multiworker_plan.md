# Loader Multi-Worker Re-architecture Plan

## Background and Pain Points
- The current loader streams data from Plex with `_iter_from_plex` by fetching entire library lists, then augments every item serially inside a single coroutine before yielding results.【F:mcp_plex/loader/__init__.py†L646-L716】
- `run` consumes that iterator, constructs document payloads, buffers them locally, and only then schedules Qdrant upserts, meaning Plex/TMDB/IMDb calls are coupled to payload creation and Qdrant throughput.【F:mcp_plex/loader/__init__.py†L842-L1096】
- Because the augmentation and Qdrant upsert preparation all happen on the main async task, we cannot overlap Plex fetches, metadata enrichment, and Qdrant writes; large libraries therefore block on the slowest stage.【F:mcp_plex/loader/__init__.py†L959-L1096】

## Goals
1. Decouple Plex ingestion, metadata enrichment, and Qdrant upserts so they can execute concurrently.
2. Make chunk sizes configurable for both Plex retrieval and metadata augmentation (default: 100 items per augmentation batch, user-configurable).
3. Preserve existing retry/backoff behavior for IMDb and Qdrant while making the pipeline resilient to worker failures.
4. Keep the CLI surface area stable while enabling future extensions (e.g., additional enrichment sources).

## Proposed Architecture

### Overview
Implement a pipeline coordinated by a new `LoaderPipeline` (name TBD) composed of three asynchronous worker roles communicating through bounded queues:

1. **Ingestion Worker**
   - Reads Plex items in configurable chunk sizes (e.g., `plex_fetch_chunk_size`).
   - Pushes raw Plex `PlexPartialObject` batches onto an `asyncio.Queue` for enrichment.
   - Continues iterating while downstream queues have capacity to avoid over-fetching.

2. **Enrichment Workers**
   - Consume Plex batches, flatten to groups of 100 items (configurable `enrichment_batch_size`).
   - For each group, perform IMDb/TMDB augmentation using existing helper functions (`_fetch_imdb_batch`, `_fetch_tmdb_*`) and produce fully constructed `AggregatedItem` objects plus the pre-built embedding payload (`models.PointStruct`).
   - Push completed points onto the upsert queue, allowing the ingestion worker to continue fetching even while metadata calls are in flight.
   - Use a worker pool (e.g., N tasks) to parallelize network-bound metadata lookups while honoring batch constraints.

3. **Upsert Worker(s)**
   - Reuse the existing `_upsert_worker` logic but move it into the pipeline coordinator so it only depends on the queue of ready-to-write points.
   - Continue supporting multiple concurrent upsert tasks with the configured semaphore and retry queue.

### Data Flow and Queues
- `ingest_queue`: carries raw Plex objects grouped by Plex chunk size. Bounded to avoid memory blow-up.
- `enriched_queue`: carries lists of `models.PointStruct` (100 per chunk by default) ready for Qdrant.
- Each queue carries a sentinel (e.g., `None`) to signal completion and allow graceful shutdown.
- Worker cancellation: propagate first failure via an `asyncio.Event`/`ExceptionGroup` equivalent; drain queues with sentinels to unblock other tasks.

### Configuration Changes
- Extend CLI/options with:
  - `--plex-chunk-size`: default 200 (tunable) describing how many Plex rating keys to fetch per batch.
  - `--enrichment-batch-size`: default 100 to match requirement.
  - Optional `--enrichment-workers`: default 4 to control metadata worker concurrency.
- Preserve existing options (e.g., upsert buffer size) and ensure new defaults keep behavior similar to current sequential pipeline when set to 1 worker.

### API/Structure Adjustments
- Extract current augmentation logic (`items.append`, payload creation, queueing) from `run` into reusable functions:
  - `build_point(item, dense_model_name, sparse_model_name)` to encapsulate text/payload assembly so enrichment workers can call it independently.【F:mcp_plex/loader/__init__.py†L959-L1063】
  - Helper to emit Qdrant batches, moving buffer management from `run` into enrichment workers to keep stage boundaries clean.
- Split `_iter_from_plex` so that raw Plex fetching and metadata fetching are distinct; the ingestion worker should focus only on retrieving Plex data (rating keys + `fetchItems`) and push them downstream before enrichment occurs.【F:mcp_plex/loader/__init__.py†L646-L716】

## Implementation Steps
1. **Refactor Utilities**
   - Extract payload/vector construction into standalone function(s) with unit tests covering movies and episodes using sample data.
   - Factor IMDb/TMDB batch helpers to accept plain IDs/objects independent of global state to simplify worker reuse.

2. **Introduce Pipeline Coordinator**
   - Create a `LoaderPipeline` (or similar) class responsible for queue setup, worker lifecycle, and error propagation (context manager to ensure graceful teardown).
   - Move IMDb cache initialization, Qdrant client setup, and retry queue persistence into pipeline initialization, reusing existing logic.【F:mcp_plex/loader/__init__.py†L864-L1106】

3. **Implement Ingestion Worker**
   - Iterate Plex sections with configurable chunk sizes; push `PlexPartialObject` batches to the enrichment queue instead of awaiting metadata immediately.
   - For sample mode, synthesize similar batches from local JSON to keep code paths consistent.

4. **Implement Enrichment Worker Pool**
   - Each worker consumes ingestion batches, performs metadata lookups in sub-batches of 100, builds `AggregatedItem` + Qdrant points, and pushes completed point batches to the upsert queue.
   - Share an `httpx.AsyncClient` per worker or per pipeline (with connection pooling) and reuse existing retry/backoff logic for IMDb.

5. **Reuse/Adapt Upsert Workers**
   - Keep `_upsert_in_batches` and retry queue unchanged, but have workers read from the enriched queue rather than managing their own buffers.
   - Ensure backpressure by awaiting `queue.join()` semantics before pipeline shutdown.

6. **Testing & Observability**
   - Add unit tests covering pipeline flow with mocked queues ensuring batches flow through all stages.
   - For integration tests, run loader against sample data to assert the same JSON output as before, verifying determinism.
   - Expand logging: include per-stage throughput metrics (items/sec) and queue sizes for troubleshooting.

## Open Questions / Follow-Ups
- Do we need rate limiting for TMDb/IMDb beyond current retry logic? Consider adding token bucket middleware if APIs enforce stricter quotas.
- Should we persist partially enriched items if Qdrant failures persist beyond retries? Potential future enhancement to dump failed batches for requeueing.
- Explore replacing global IMDb/TMDB settings with dependency-injected objects for improved testability and easier worker reuse.
