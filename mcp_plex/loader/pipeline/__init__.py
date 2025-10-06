"""Expose the concrete loader pipeline stages and shared channel helpers."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .channels import (
    EpisodeBatch,
    IMDbRetryQueue,
    IngestBatch,
    IngestQueue,
    INGEST_DONE,
    MovieBatch,
    PERSIST_DONE,
    PersistenceQueue,
    SampleBatch,
    chunk_sequence,
    require_positive,
)

if TYPE_CHECKING:
    from .enrichment import EnrichmentStage
    from .ingestion import IngestionStage
    from .orchestrator import LoaderOrchestrator
    from .persistence import PersistenceStage

__all__ = [
    "IngestionStage",
    "EnrichmentStage",
    "PersistenceStage",
    "LoaderOrchestrator",
    "MovieBatch",
    "EpisodeBatch",
    "SampleBatch",
    "IngestBatch",
    "IngestQueue",
    "PersistenceQueue",
    "INGEST_DONE",
    "PERSIST_DONE",
    "IMDbRetryQueue",
    "chunk_sequence",
    "require_positive",
]

_STAGE_MODULES = {
    "IngestionStage": ".ingestion",
    "EnrichmentStage": ".enrichment",
    "PersistenceStage": ".persistence",
    "LoaderOrchestrator": ".orchestrator",
}


def __getattr__(name: str) -> Any:
    """Lazily import pipeline stage classes on first access."""

    if name in _STAGE_MODULES:
        module = import_module(f"{__name__}{_STAGE_MODULES[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return module attributes for introspection tools."""

    return sorted(set(globals()) | set(__all__))
