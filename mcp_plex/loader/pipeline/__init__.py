"""Loader pipeline package exports placeholder interfaces for pipeline stages."""

from __future__ import annotations

from typing import Protocol


class ChannelDefinition(Protocol):
    """Placeholder protocol for defining loader pipeline channels."""


class IngestionStage(Protocol):
    """Placeholder protocol for pipeline ingestion stage implementations."""


class EnrichmentStage(Protocol):
    """Placeholder protocol for pipeline enrichment stage implementations."""


class PersistenceStage(Protocol):
    """Placeholder protocol for pipeline persistence stage implementations."""


class PipelineOrchestrator(Protocol):
    """Placeholder protocol for orchestrating loader pipeline stages."""


__all__ = [
    "ChannelDefinition",
    "IngestionStage",
    "EnrichmentStage",
    "PersistenceStage",
    "PipelineOrchestrator",
]
