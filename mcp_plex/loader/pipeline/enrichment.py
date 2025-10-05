"""Enrichment stage coordinator for the loader pipeline.

Movie metadata enrichment has been ported from the legacy loader and now
performs TMDb and IMDb lookups before emitting aggregated payloads to the
persistence queue.  Episode enrichment reuses the TMDb caching and lookup
logic, and sample-mode batches pass straight through to persistence so end to
end processing mirrors the legacy worker implementation.
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

from ...common.types import AggregatedItem, TMDBEpisode, TMDBItem, TMDBShow
from .. import (
    _build_plex_item,
    _extract_external_ids,
    _fetch_imdb_batch,
    _fetch_tmdb_episode,
    _fetch_tmdb_movie,
    _fetch_tmdb_show,
    resolve_tmdb_season_number,
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
        self._show_tmdb_cache: dict[str, TMDBShow | None] = {}

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
                self._logger.info(
                    "Processed movie batch with %d items (queue size=%d)",
                    len(aggregated),
                    self._persistence_queue.qsize(),
                )

    async def _handle_episode_batch(self, batch: EpisodeBatch) -> None:
        """Enrich and forward Plex episode batches to the persistence stage."""

        episode_chunks = [
            list(chunk)
            for chunk in chunk_sequence(batch.episodes, self._episode_batch_size)
            if len(chunk)
        ]
        if not episode_chunks:
            return

        show_title = getattr(batch.show, "title", str(batch.show))
        async with self._acquire_http_client() as client:
            for episodes in episode_chunks:
                aggregated = await self._enrich_episodes(client, batch.show, episodes)
                await self._emit_persistence_batch(aggregated)
                self._logger.info(
                    "Processed episode batch for %s with %d items (queue size=%d)",
                    show_title,
                    len(aggregated),
                    self._persistence_queue.qsize(),
                )

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
        """Forward sample data batches directly to the persistence stage."""

        for items in chunk_sequence(batch.items, self._movie_batch_size):
            aggregated = list(items)
            if not aggregated:
                continue
            await self._emit_persistence_batch(aggregated)
            self._logger.info(
                "Processed sample batch with %d items (queue size=%d)",
                len(aggregated),
                self._persistence_queue.qsize(),
            )

    async def _enrich_episodes(
        self, client: Any, show: Any, episodes: Sequence[Any]
    ) -> list[AggregatedItem]:
        """Fetch external metadata for *episodes* and aggregate the results."""

        show_ids = _extract_external_ids(show)
        show_tmdb: TMDBShow | None = None
        if show_ids.tmdb:
            show_tmdb = await self._get_tmdb_show(client, show_ids.tmdb)
        episode_ids = [_extract_external_ids(ep) for ep in episodes]
        imdb_ids = [ids.imdb for ids in episode_ids if ids.imdb]
        imdb_map = (
            await _fetch_imdb_batch(client, imdb_ids) if imdb_ids else {}
        )

        tmdb_results: list[TMDBEpisode | None] = []
        if show_tmdb:
            episode_tasks = [
                self._lookup_tmdb_episode(client, show_tmdb, ep)
                for ep in episodes
            ]
            if episode_tasks:
                tmdb_results = await asyncio.gather(*episode_tasks)
        tmdb_iter = iter(tmdb_results)

        aggregated: list[AggregatedItem] = []
        for ep, ids in zip(episodes, episode_ids):
            tmdb_episode = next(tmdb_iter, None) if show_tmdb else None
            imdb = imdb_map.get(ids.imdb) if ids.imdb else None
            tmdb_item: TMDBItem | None = tmdb_episode or show_tmdb
            aggregated.append(
                AggregatedItem(
                    plex=_build_plex_item(ep),
                    imdb=imdb,
                    tmdb=tmdb_item,
                )
            )
        return aggregated

    async def _get_tmdb_show(
        self, client: Any, tmdb_id: str
    ) -> TMDBShow | None:
        """Return the TMDb show for *tmdb_id*, using the in-memory cache."""

        tmdb_id_str = str(tmdb_id)
        if tmdb_id_str in self._show_tmdb_cache:
            return self._show_tmdb_cache[tmdb_id_str]
        show = await _fetch_tmdb_show(
            client, tmdb_id_str, self._tmdb_api_key or ""
        )
        self._show_tmdb_cache[tmdb_id_str] = show
        return show

    async def _lookup_tmdb_episode(
        self, client: Any, show_tmdb: TMDBShow | None, episode: Any
    ) -> TMDBEpisode | None:
        """Lookup the TMDb metadata for *episode* within *show_tmdb*."""

        if not show_tmdb or not self._tmdb_api_key:
            return None
        season = resolve_tmdb_season_number(show_tmdb, episode)
        ep_num = getattr(episode, "index", None)
        if isinstance(ep_num, str) and ep_num.isdigit():
            ep_num = int(ep_num)
        if season is None or ep_num is None:
            return None
        return await _fetch_tmdb_episode(
            client,
            show_tmdb.id,
            season,
            ep_num,
            self._tmdb_api_key,
        )
