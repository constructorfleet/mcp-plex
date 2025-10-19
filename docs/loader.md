# Loader Operations

The loader emits structured log records throughout ingestion so operators can
monitor progress without scraping stdout dumps.

## Qdrant retry summary

After every run the loader processes the in-memory Qdrant retry queue and
reports the outcome via a structured ``INFO`` log:

```
Qdrant retry summary: 123 succeeded, 4 failed
```

The log record is tagged with ``event="qdrant_retry_summary"`` and includes two
integer attributes:

- ``succeeded_points`` – number of points that were reindexed successfully after
  retrying. The human-readable message echoes this value for quick scanning.
- ``failed_points`` – number of points that still failed after exhausting all
  retry attempts and therefore remain missing from the collection. This value is
  also repeated in the message so dashboards can surface it without structured
  parsing.

Use your logging aggregator or ``caplog`` when testing to filter on the
``qdrant_retry_summary`` event and confirm ingestion health. Search for
``"Qdrant retry summary"`` in log streams to see the summarized counts when
structured metadata is unavailable.
