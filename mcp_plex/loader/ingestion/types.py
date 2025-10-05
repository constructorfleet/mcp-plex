from __future__ import annotations

from dataclasses import dataclass
from typing import Union

try:
    from plexapi.base import PlexPartialObject
except Exception:  # pragma: no cover - fall back when plexapi unavailable
    PlexPartialObject = object  # type: ignore[assignment]

from mcp_plex.common.types import AggregatedItem


@dataclass(slots=True)
class MovieBatch:
    """Batch of Plex movie items pending metadata enrichment."""

    movies: list[PlexPartialObject]


@dataclass(slots=True)
class EpisodeBatch:
    """Batch of Plex episodes along with their parent show."""

    show: PlexPartialObject
    episodes: list[PlexPartialObject]


@dataclass(slots=True)
class SampleBatch:
    """Batch of pre-enriched items used by sample mode."""

    items: list[AggregatedItem]


IngestBatch = Union[MovieBatch, EpisodeBatch, SampleBatch]

__all__ = [
    "EpisodeBatch",
    "IngestBatch",
    "MovieBatch",
    "SampleBatch",
]
