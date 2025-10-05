"""Batch container and helper utilities shared across loader stages."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence, TypeVar

from ...common.types import AggregatedItem

try:  # Only import plexapi when available; the sample data mode does not require it.
    from plexapi.base import PlexPartialObject
except Exception:
    PlexPartialObject = object  # type: ignore[assignment]

T = TypeVar("T")


@dataclass(slots=True)
class MovieBatch:
    """Batch of Plex movie items pending metadata enrichment."""

    movies: list["PlexPartialObject"]


@dataclass(slots=True)
class EpisodeBatch:
    """Batch of Plex episodes along with their parent show."""

    show: "PlexPartialObject"
    episodes: list["PlexPartialObject"]


@dataclass(slots=True)
class SampleBatch:
    """Batch of pre-enriched items used by sample mode."""

    items: list[AggregatedItem]


IngestBatch = MovieBatch | EpisodeBatch | SampleBatch


def require_positive(value: int, *, name: str) -> int:
    """Return *value* if positive, otherwise raise a ``ValueError``."""

    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def chunk_sequence(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield ``items`` in chunks of at most ``size`` elements."""

    size = require_positive(int(size), name="size")
    for start in range(0, len(items), size):
        yield items[start : start + size]


class IMDbRetryQueue(asyncio.Queue[str]):
    """Queue that tracks items in a deque for safe serialization."""

    def __init__(self, initial: Iterable[str] | None = None):
        super().__init__()
        self._items: deque[str] = deque()
        if initial:
            for imdb_id in initial:
                imdb_id_str = str(imdb_id)
                super().put_nowait(imdb_id_str)
                self._items.append(imdb_id_str)

    def put_nowait(self, item: str) -> None:  # type: ignore[override]
        super().put_nowait(item)
        self._items.append(item)

    def get_nowait(self) -> str:  # type: ignore[override]
        if not self._items:
            raise RuntimeError(
                "Desynchronization: Queue is not empty but self._items is empty."
            )
        try:
            item = super().get_nowait()
        except asyncio.QueueEmpty:
            raise RuntimeError(
                "Desynchronization: self._items is not empty but asyncio.Queue is empty."
            )
        self._items.popleft()
        return item

    def snapshot(self) -> list[str]:
        """Return a list of the current queue contents."""

        return list(self._items)


# Backwards-compatible aliases for private imports while callers migrate.
_MovieBatch = MovieBatch
_EpisodeBatch = EpisodeBatch
_SampleBatch = SampleBatch
_IngestBatch = IngestBatch
_require_positive = require_positive
_chunk_sequence = chunk_sequence
_IMDbRetryQueue = IMDbRetryQueue

__all__ = [
    "MovieBatch",
    "EpisodeBatch",
    "SampleBatch",
    "IngestBatch",
    "require_positive",
    "chunk_sequence",
    "IMDbRetryQueue",
    "_MovieBatch",
    "_EpisodeBatch",
    "_SampleBatch",
    "_IngestBatch",
    "_require_positive",
    "_chunk_sequence",
    "_IMDbRetryQueue",
]
