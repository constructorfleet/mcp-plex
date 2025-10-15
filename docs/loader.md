# Loader Operations Guide

## Monitoring ingestion results

The staged loader logs a structured summary after it processes the Qdrant retry
queue. Operators can filter for the `Qdrant retry summary` message to see how
many points were successfully retried and how many were dropped after exhausting
all attempts. The log record exposes the counts as structured attributes:

- `qdrant_retry_succeeded`: number of points upserted during retry handling.
- `qdrant_retry_failed`: number of points that still failed after all retries.

These attributes are attached to the log record (via `extra`) so they are
available to structured logging backends and observability pipelines without
additional parsing.
