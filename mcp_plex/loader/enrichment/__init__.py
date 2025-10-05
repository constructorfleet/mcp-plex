from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Sequence

import httpx
from qdrant_client import models

from mcp_plex.common.types import AggregatedItem, TMDBShow

from ..imdb_cache import IMDbCache
from ..ingestion.utils import chunk_sequence
from ..utils import require_positive
from ..storage.utils import build_point
from .types import IMDbRetryQueue
from .utils import (
    enrich_episodes,
    enrich_movies,
)

try:
    from plexapi.base import PlexPartialObject
except Exception:  # pragma: no cover - fall back when plexapi unavailable
    PlexPartialObject = object  # type: ignore[assignment]


class EnrichmentTask:
    """Enrich Plex metadata and emit Qdrant points for storage."""

    def __init__(
        self,
        ingest_queue: asyncio.Queue[object | None],
        points_queue: asyncio.Queue[list[models.PointStruct] | None],
        *,
        http_client: httpx.AsyncClient,
        tmdb_api_key: str | None,
        imdb_cache: IMDbCache | None,
        imdb_retry_queue: IMDbRetryQueue,
        imdb_batch_limit: int,
        imdb_max_retries: int,
        imdb_backoff: float,
        dense_model_name: str,
        sparse_model_name: str,
        enrichment_batch_size: int,
        worker_count: int,
        upsert_buffer_size: int,
        upsert_capacity: asyncio.Semaphore,
        log_progress: Callable[[str, int, float, int], None],
        logger: logging.Logger,
    ) -> None:
        self._ingest_queue = ingest_queue
        self._points_queue = points_queue
        self._http_client = http_client
        self._tmdb_api_key = tmdb_api_key
        self._imdb_cache = imdb_cache
        self._imdb_retry_queue = imdb_retry_queue
        self._imdb_batch_limit = imdb_batch_limit
        self._imdb_max_retries = imdb_max_retries
        self._imdb_backoff = imdb_backoff
        self._dense_model_name = dense_model_name
        self._sparse_model_name = sparse_model_name
        self._enrichment_batch_size = require_positive(
            enrichment_batch_size, name="enrichment_batch_size"
        )
        self._worker_count = require_positive(worker_count, name="worker_count")
        self._upsert_buffer_size = require_positive(
            upsert_buffer_size, name="upsert_buffer_size"
        )
        self._upsert_capacity = upsert_capacity
        self._log_progress = log_progress
        self._logger = logger

        self._items: list[AggregatedItem] = []
        self._show_tmdb_cache: dict[str, TMDBShow | None] = {}
        self._enriched_count = 0
        self._enrich_start = time.perf_counter()

    @property
    def items(self) -> list[AggregatedItem]:
        return self._items

    def start_workers(self) -> list[asyncio.Task[None]]:
        return [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self._worker_count)
        ]

    async def _worker(self, worker_id: int) -> None:
        while True:
            batch = await self._ingest_queue.get()
            if batch is None:
                self._ingest_queue.task_done()
                break
            try:
                await self._process_batch(batch)
            finally:
                self._ingest_queue.task_done()

    async def _process_batch(self, batch: object) -> None:
        from ..ingestion.types import EpisodeBatch, MovieBatch, SampleBatch

        if isinstance(batch, MovieBatch):
            await self._process_movie_batch(batch)
        elif isinstance(batch, EpisodeBatch):
            await self._process_episode_batch(batch)
        elif isinstance(batch, SampleBatch):
            await self._process_sample_batch(batch)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    async def _process_movie_batch(self, batch: object) -> None:
        from ..ingestion.types import MovieBatch

        assert isinstance(batch, MovieBatch)
        for chunk in chunk_sequence(batch.movies, self._enrichment_batch_size):
            movies = list(chunk)
            if not movies:
                continue
            await self._enrich_movies(movies)

    async def _process_episode_batch(self, batch: object) -> None:
        from ..ingestion.types import EpisodeBatch

        assert isinstance(batch, EpisodeBatch)
        for chunk in chunk_sequence(batch.episodes, self._enrichment_batch_size):
            episodes = list(chunk)
            if not episodes:
                continue
            await self._enrich_episodes(batch.show, episodes)

    async def _process_sample_batch(self, batch: object) -> None:
        from ..ingestion.types import SampleBatch

        assert isinstance(batch, SampleBatch)
        for chunk in chunk_sequence(batch.items, self._enrichment_batch_size):
            aggregated = list(chunk)
            if not aggregated:
                continue
            await self._emit_points(aggregated)

    async def _enrich_movies(self, movies: Sequence[PlexPartialObject]) -> None:
        aggregated = await enrich_movies(
            self._http_client,
            movies,
            tmdb_api_key=self._tmdb_api_key,
            imdb_cache=self._imdb_cache,
            imdb_batch_limit=self._imdb_batch_limit,
            imdb_max_retries=self._imdb_max_retries,
            imdb_backoff=self._imdb_backoff,
            imdb_retry_queue=self._imdb_retry_queue,
            logger=self._logger,
        )
        await self._emit_points(aggregated)

    async def _enrich_episodes(
        self, show: PlexPartialObject, episodes: Sequence[PlexPartialObject]
    ) -> None:
        aggregated = await enrich_episodes(
            self._http_client,
            show,
            episodes,
            tmdb_api_key=self._tmdb_api_key,
            imdb_cache=self._imdb_cache,
            imdb_batch_limit=self._imdb_batch_limit,
            imdb_max_retries=self._imdb_max_retries,
            imdb_backoff=self._imdb_backoff,
            imdb_retry_queue=self._imdb_retry_queue,
            show_tmdb_cache=self._show_tmdb_cache,
            logger=self._logger,
        )
        await self._emit_points(aggregated)

    async def _emit_points(self, aggregated: Sequence[AggregatedItem]) -> None:
        if not aggregated:
            return
        if self._enriched_count == 0:
            self._enrich_start = time.perf_counter()
        self._items.extend(aggregated)
        self._enriched_count += len(aggregated)
        self._log_progress(
            "Enrichment",
            self._enriched_count,
            self._enrich_start,
            self._points_queue.qsize(),
        )
        points = [
            build_point(item, self._dense_model_name, self._sparse_model_name)
            for item in aggregated
        ]
        for chunk in chunk_sequence(points, self._upsert_buffer_size):
            batch = list(chunk)
            if not batch:
                continue
            await self._upsert_capacity.acquire()
            try:
                await self._points_queue.put(batch)
            except BaseException:
                self._upsert_capacity.release()
                raise


__all__ = ["EnrichmentTask", "IMDbRetryQueue"]
