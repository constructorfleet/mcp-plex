"""Batch container and helper utilities shared across loader stages.

The queues exported here intentionally share sentinel objects so stage-specific
integration tests can assert on hand-off behavior without duplicating
constants.  The loader still emits ``None`` as a completion token for
compatibility while downstream components migrate to sentinel-only signaling.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Final,
    Iterable,
    Literal,
    Sequence,
    TypeAlias,
    TypeVar,
)

from ...common.types import AggregatedItem
from ...common.validation import require_positive

from plexapi.video import Episode, Movie, Show

T = TypeVar("T")

if TYPE_CHECKING:
    from qdrant_client import models


INGEST_DONE: Final = object()
IngestSentinel: TypeAlias = Literal[INGEST_DONE]
"""Sentinel object signaling that ingestion has completed.

The loader currently places ``None`` on ingestion queues in addition to this
sentinel so legacy listeners that only check for ``None`` continue to work.
"""

PERSIST_DONE: Final = object()
PersistenceSentinel: TypeAlias = Literal[PERSIST_DONE]
"""Sentinel object signaling that persistence has completed."""

if TYPE_CHECKING:
    PersistencePayload: TypeAlias = list[models.PointStruct]

PersistencePayload: TypeAlias = list["models.PointStruct"]


@dataclass(slots=True)
class MovieBatch:
    """Batch of Plex movie items pending metadata enrichment."""

    movies: list["Movie"]


@dataclass(slots=True)
class EpisodeBatch:
    """Batch of Plex episodes along with their parent show."""

    show: "Show"
    episodes: list["Episode"]


@dataclass(slots=True)
class SampleBatch:
    """Batch of pre-enriched items used by sample mode."""

    items: list[AggregatedItem]


IngestBatch = MovieBatch | EpisodeBatch | SampleBatch

IngestQueueItem: TypeAlias = IngestBatch | None | IngestSentinel
PersistenceQueueItem: TypeAlias = (
    PersistencePayload | None | PersistenceSentinel
)

IngestQueue: TypeAlias = asyncio.Queue[IngestQueueItem]
PersistenceQueue: TypeAlias = asyncio.Queue[PersistenceQueueItem]


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


async def enqueue_nowait(queue: asyncio.Queue[T], item: T) -> None:
    """Place *item* onto *queue* using ``put_nowait`` with fallback backpressure."""

    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        await queue.put(item)


__all__ = [
    "MovieBatch",
    "EpisodeBatch",
    "SampleBatch",
    "IngestBatch",
    "INGEST_DONE",
    "IngestSentinel",
    "PERSIST_DONE",
    "PersistenceSentinel",
    "IngestQueue",
    "PersistenceQueue",
    "chunk_sequence",
    "IMDbRetryQueue",
    "enqueue_nowait",
]
