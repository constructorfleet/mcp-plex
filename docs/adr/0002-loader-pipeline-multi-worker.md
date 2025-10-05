# ADR 0002: Loader Multi-Worker Pipeline

## Status
Accepted

## Context
The Plex loader previously executed ingestion, enrichment, and Qdrant writes sequentially. Each phase waited for the previous phase to finish, so TMDb and IMDb lookups, embedding generation, and upsert requests could not overlap. Large libraries therefore spent most of their time idle while waiting on network latency, and any failure in a downstream phase required rerunning the entire load.

Stakeholders asked for the "loader multi worker rearchitecture plan" to overlap these phases, provide better fault isolation, and expose concurrency controls. The new design must also integrate with existing logging, testing, and configuration patterns.

## Decision
We introduced a `LoaderPipeline` that streams points through dedicated stages backed by bounded queues:

1. An ingestion worker pulls Plex items, resolves TMDb and IMDb metadata, and enqueues normalized `LoaderItem` batches.
2. A configurable pool of enrichment workers generates dense and sparse embeddings in parallel and enqueues `PointBatch` payloads.
3. An upsert worker sends Qdrant `upsert` requests as soon as batches are ready, logging partial failures without stalling upstream work.

The pipeline propagates fatal errors across stages, drains queues on shutdown, records per-stage throughput, and surfaces CLI options so operators can tune Plex chunk size, enrichment batch size, and worker counts.

## Consequences
- Loader runs overlap I/O and compute, reducing end-to-end latency for large libraries.
- Failure handling is localized: individual batch errors are logged and skipped while fatal exceptions halt the entire pipeline predictably.
- Logging, tests, and documentation must reflect the staged pipeline model and validate concurrency behaviors.
- Operators can experiment with worker counts and batch sizes without code changes.

## Implementation Notes
- The pipeline and supporting dataclasses live in `mcp_plex/loader/__init__.py` with unit tests in `tests/test_loader_unit.py` and logging coverage in `tests/test_loader_logging.py`.
- CLI defaults align with the previous sequential behavior so existing deployments continue to work without extra configuration.
- Version metadata in `pyproject.toml`, `docker/pyproject.deps.toml`, and `uv.lock` is kept in sync whenever loader architecture changes are shipped.
