from __future__ import annotations

import asyncio
import time
from typing import Callable, Sequence

from mcp_plex.common.types import AggregatedItem

from ..utils import require_positive
from .types import EpisodeBatch, IngestBatch, MovieBatch, SampleBatch
from .utils import chunk_sequence

try:
    from plexapi.server import PlexServer
except Exception:  # pragma: no cover - fall back when plexapi unavailable
    PlexServer = None  # type: ignore[assignment]


class IngestionTask:
    """Fetch Plex data and feed ingestion batches to the enrichment stage."""

    def __init__(
        self,
        queue: asyncio.Queue[IngestBatch | None],
        *,
        sample_items: list[AggregatedItem] | None,
        plex_server: PlexServer | None,
        plex_chunk_size: int,
        enrichment_batch_size: int,
        enrichment_workers: int,
        log_progress: Callable[[str, int, float, int], None],
    ) -> None:
        self._queue = queue
        self._sample_items = sample_items
        self._server = plex_server
        self._plex_chunk_size = require_positive(plex_chunk_size, name="plex_chunk_size")
        self._enrichment_batch_size = require_positive(
            enrichment_batch_size, name="enrichment_batch_size"
        )
        self._worker_count = require_positive(
            enrichment_workers, name="enrichment_workers"
        )
        self._log_progress = log_progress
        self._count = 0
        self._start = time.perf_counter()

    @property
    def count(self) -> int:
        return self._count

    @property
    def start_time(self) -> float:
        return self._start

    async def run(self) -> None:
        self._start = time.perf_counter()
        try:
            if self._sample_items is not None:
                await self._ingest_sample(self._sample_items)
            else:
                await self._ingest_from_plex()
        finally:
            for _ in range(self._worker_count):
                await self._queue.put(None)
            self._log_progress("Ingestion", self._count, self._start, self._queue.qsize())

    async def _ingest_sample(self, items: Sequence[AggregatedItem]) -> None:
        for chunk in chunk_sequence(items, self._enrichment_batch_size):
            batch = SampleBatch(items=list(chunk))
            if not batch.items:
                continue
            await self._queue.put(batch)
            self._count += len(batch.items)
            self._log_progress(
                "Ingestion",
                self._count,
                self._start,
                self._queue.qsize(),
            )

    async def _ingest_from_plex(self) -> None:
        if self._server is None:
            raise RuntimeError("Plex server unavailable for ingestion")
        movie_section = self._server.library.section("Movies")
        movie_keys = [int(m.ratingKey) for m in movie_section.all()]
        for key_chunk in chunk_sequence(movie_keys, self._plex_chunk_size):
            key_list = list(key_chunk)
            movies = list(self._server.fetchItems(key_list)) if key_list else []
            if not movies:
                continue
            await self._queue.put(MovieBatch(movies=movies))
            self._count += len(movies)
            self._log_progress(
                "Ingestion",
                self._count,
                self._start,
                self._queue.qsize(),
            )
        show_section = self._server.library.section("TV Shows")
        show_keys = [int(s.ratingKey) for s in show_section.all()]
        for show_chunk in chunk_sequence(show_keys, self._plex_chunk_size):
            shows = list(self._server.fetchItems(list(show_chunk)))
            for show in shows:
                episode_keys = [int(e.ratingKey) for e in show.episodes()]
                for episode_chunk in chunk_sequence(
                    episode_keys, self._plex_chunk_size
                ):
                    keys = list(episode_chunk)
                    episodes = list(self._server.fetchItems(keys)) if keys else []
                    if not episodes:
                        continue
                    await self._queue.put(EpisodeBatch(show=show, episodes=episodes))
                    self._count += len(episodes)
                    self._log_progress(
                        "Ingestion",
                        self._count,
                        self._start,
                        self._queue.qsize(),
                    )


__all__ = ["IngestionTask", "EpisodeBatch", "IngestBatch", "MovieBatch", "SampleBatch"]
