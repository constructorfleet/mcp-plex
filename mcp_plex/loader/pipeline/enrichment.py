"""Enrichment stage coordinator for the loader pipeline.

Movie metadata enrichment has been ported from the legacy loader and now
performs TMDb and IMDb lookups before emitting aggregated payloads to the
persistence queue.  Episode and sample handling remain placeholder hooks while
the rest of the legacy logic is migrated.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
import inspect
from typing import Any

from .channels import (
    EpisodeBatch,
    IMDbRetryQueue,
    INGEST_DONE,
    IngestQueue,
    MovieBatch,
    PersistenceQueue,
    SampleBatch,
    chunk_sequence,
    require_positive,
)

from ...common.types import AggregatedItem
from .. import (
    _build_plex_item,
    _extract_external_ids,
    _fetch_imdb_batch,
    _fetch_tmdb_movie,
)


class EnrichmentStage:
    """Coordinate metadata enrichment for ingested media batches."""

    def __init__(
        self,
        *,
        http_client_factory: Callable[[], Awaitable[Any] | Any],
        tmdb_api_key: str,
        ingest_queue: IngestQueue,
        persistence_queue: PersistenceQueue,
        imdb_retry_queue: IMDbRetryQueue | None,
        movie_batch_size: int,
        episode_batch_size: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self._http_client_factory = http_client_factory
        self._tmdb_api_key = str(tmdb_api_key)
        self._ingest_queue = ingest_queue
        self._persistence_queue = persistence_queue
        self._imdb_retry_queue = imdb_retry_queue or IMDbRetryQueue()
        requested_movie_batch_size = require_positive(
            int(movie_batch_size), name="movie_batch_size"
        )
        self._movie_batch_size = min(requested_movie_batch_size, 100)
        self._episode_batch_size = require_positive(
            int(episode_batch_size), name="episode_batch_size"
        )
        self._logger = logger or logging.getLogger("mcp_plex.loader.enrichment")

    @property
    def logger(self) -> logging.Logger:
        """Logger used by the enrichment stage."""

        return self._logger

    @property
    def imdb_retry_queue(self) -> IMDbRetryQueue:
        """IMDb retry queue used by the enrichment stage."""

        return self._imdb_retry_queue

    async def run(self) -> None:
        """Execute the enrichment stage."""

        while True:
            batch = await self._ingest_queue.get()
            try:
                if batch is None:
                    self._logger.debug(
                        "Received legacy completion token; ignoring."
                    )
                    continue

                if batch is INGEST_DONE:
                    self._logger.info(
                        "Ingestion completed; finishing enrichment stage."
                    )
                    break

                if isinstance(batch, MovieBatch):
                    await self._handle_movie_batch(batch)
                elif isinstance(batch, EpisodeBatch):
                    await self._handle_episode_batch(batch)
                elif isinstance(batch, SampleBatch):
                    await self._handle_sample_batch(batch)
                else:  # pragma: no cover - defensive logging for future types
                    self._logger.warning(
                        "Received unsupported batch type: %r", batch
                    )
            finally:
                self._ingest_queue.task_done()

        await self._persistence_queue.put(None)

    async def _handle_movie_batch(self, batch: MovieBatch) -> None:
        """Enrich and forward Plex movie batches to the persistence stage."""

        movie_chunks = [
            list(chunk)
            for chunk in chunk_sequence(batch.movies, self._movie_batch_size)
            if len(chunk)
        ]
        if not movie_chunks:
            return

        async with self._acquire_http_client() as client:
            for movies in movie_chunks:
                aggregated = await self._enrich_movies(client, movies)
                await self._emit_persistence_batch(aggregated)

    async def _handle_episode_batch(self, batch: EpisodeBatch) -> None:
        """Placeholder hook for processing Plex episode batches."""

        show_title = getattr(batch.show, "title", str(batch.show))
        episode_count = len(batch.episodes)
        self._logger.info(
            "Episode enrichment has not been ported yet; %d episodes pending for %s.",
            episode_count,
            show_title,
        )
        await asyncio.sleep(0)

    @asynccontextmanager
    async def _acquire_http_client(self) -> AsyncIterator[Any]:
        """Yield an HTTP client from the injected factory."""

        resource = self._http_client_factory()
        if inspect.isawaitable(resource):
            resource = await resource

        if hasattr(resource, "__aenter__") and hasattr(resource, "__aexit__"):
            async with resource as client:
                yield client
            return

        if hasattr(resource, "__enter__") and hasattr(resource, "__exit__"):
            with resource as client:
                yield client
            return

        try:
            yield resource
        finally:
            closer = getattr(resource, "aclose", None)
            if callable(closer):
                result = closer()
                if inspect.isawaitable(result):
                    await result
                return
            closer = getattr(resource, "close", None)
            if callable(closer):
                result = closer()
                if inspect.isawaitable(result):
                    await result

    async def _emit_persistence_batch(
        self, aggregated: Sequence[AggregatedItem]
    ) -> None:
        """Place aggregated items onto the persistence queue."""

        if not aggregated:
            return
        await self._persistence_queue.put(list(aggregated))

    async def _enrich_movies(
        self, client: Any, movies: Sequence[Any]
    ) -> list[AggregatedItem]:
        """Fetch external metadata for *movies* and aggregate the results."""

        movie_ids = [_extract_external_ids(movie) for movie in movies]
        imdb_ids = [ids.imdb for ids in movie_ids if ids.imdb]
        imdb_map = (
            await _fetch_imdb_batch(client, imdb_ids) if imdb_ids else {}
        )

        tmdb_results: list[Any] = []
        api_key = self._tmdb_api_key
        if api_key:
            tmdb_tasks = [
                _fetch_tmdb_movie(client, ids.tmdb, api_key)
                for ids in movie_ids
                if ids.tmdb
            ]
            if tmdb_tasks:
                tmdb_results = await asyncio.gather(*tmdb_tasks)
        tmdb_iter = iter(tmdb_results)

        aggregated: list[AggregatedItem] = []
        for movie, ids in zip(movies, movie_ids):
            tmdb = next(tmdb_iter, None) if ids.tmdb else None
            imdb = imdb_map.get(ids.imdb) if ids.imdb else None
            aggregated.append(
                AggregatedItem(
                    plex=_build_plex_item(movie),
                    imdb=imdb,
                    tmdb=tmdb,
                )
            )
        return aggregated

    async def _handle_sample_batch(self, batch: SampleBatch) -> None:
        """Placeholder hook for processing sample data batches."""

        item_count = len(batch.items)
        self._logger.info(
            "Sample enrichment has not been ported yet; %d items queued for later.",
            item_count,
        )
        await asyncio.sleep(0)
