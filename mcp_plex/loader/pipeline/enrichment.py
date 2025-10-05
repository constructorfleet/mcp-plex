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
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
import inspect
from typing import Any

import httpx

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

from ...common.types import AggregatedItem, IMDbTitle, TMDBEpisode, TMDBItem, TMDBShow
from .. import (
    _build_plex_item,
    _extract_external_ids,
    _fetch_tmdb_episode,
    _fetch_tmdb_movie,
    _fetch_tmdb_show,
    resolve_tmdb_season_number,
)
from ..imdb_cache import IMDbCache


LOGGER = logging.getLogger(__name__)


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
        imdb_cache: IMDbCache | None = None,
        imdb_max_retries: int = 3,
        imdb_backoff: float = 1.0,
        imdb_batch_limit: int = 5,
        imdb_requests_per_window: int | None = None,
        imdb_window_seconds: float = 1.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self._http_client_factory = http_client_factory
        self._tmdb_api_key = str(tmdb_api_key)
        self._ingest_queue = ingest_queue
        self._persistence_queue = persistence_queue
        self._imdb_retry_queue = imdb_retry_queue or IMDbRetryQueue()
        self._imdb_cache = imdb_cache
        if imdb_max_retries < 0:
            raise ValueError("imdb_max_retries must be non-negative")
        if imdb_backoff < 0:
            raise ValueError("imdb_backoff must be non-negative")
        self._imdb_max_retries = int(imdb_max_retries)
        self._imdb_backoff = float(imdb_backoff)
        self._imdb_batch_limit = require_positive(
            int(imdb_batch_limit), name="imdb_batch_limit"
        )
        if imdb_requests_per_window is not None and imdb_requests_per_window <= 0:
            raise ValueError(
                "imdb_requests_per_window must be positive when provided"
            )
        if imdb_window_seconds <= 0:
            raise ValueError("imdb_window_seconds must be positive")
        self._imdb_throttle = _RequestThrottler(
            limit=imdb_requests_per_window,
            interval=float(imdb_window_seconds),
        )
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
            got_item = False
            try:
                batch = self._ingest_queue.get_nowait()
                got_item = True
            except asyncio.QueueEmpty:
                if await self._retry_imdb_batches():
                    continue
                batch = await self._ingest_queue.get()
                got_item = True
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
                if got_item:
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
            await _fetch_imdb_batch(
                client,
                imdb_ids,
                cache=self._imdb_cache,
                throttle=self._imdb_throttle,
                max_retries=self._imdb_max_retries,
                backoff=self._imdb_backoff,
                retry_queue=self._imdb_retry_queue,
                batch_limit=self._imdb_batch_limit,
            )
            if imdb_ids
            else {}
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
            await _fetch_imdb_batch(
                client,
                imdb_ids,
                cache=self._imdb_cache,
                throttle=self._imdb_throttle,
                max_retries=self._imdb_max_retries,
                backoff=self._imdb_backoff,
                retry_queue=self._imdb_retry_queue,
                batch_limit=self._imdb_batch_limit,
            )
            if imdb_ids
            else {}
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

    async def _retry_imdb_batches(self) -> bool:
        """Process IMDb IDs from the retry queue when idle."""

        if self._imdb_retry_queue.empty():
            return False

        imdb_ids: list[str] = []
        while (
            len(imdb_ids) < self._imdb_batch_limit
            and not self._imdb_retry_queue.empty()
        ):
            try:
                imdb_ids.append(self._imdb_retry_queue.get_nowait())
            except asyncio.QueueEmpty:  # pragma: no cover - race with producers
                break

        if not imdb_ids:
            return False

        async with self._acquire_http_client() as client:
            await _fetch_imdb_batch(
                client,
                imdb_ids,
                cache=self._imdb_cache,
                throttle=self._imdb_throttle,
                max_retries=self._imdb_max_retries,
                backoff=self._imdb_backoff,
                retry_queue=self._imdb_retry_queue,
                batch_limit=self._imdb_batch_limit,
            )

        self._logger.info(
            "Retried IMDb batch with %d items (retry queue=%d)",
            len(imdb_ids),
            self._imdb_retry_queue.qsize(),
        )
        return True


class _RequestThrottler:
    """Simple asynchronous rate limiter for external requests."""

    def __init__(self, *, limit: int | None, interval: float) -> None:
        self._limit = limit
        self._interval = interval
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Await until another request can be issued."""

        if self._limit is None:
            return

        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            while self._timestamps and now - self._timestamps[0] >= self._interval:
                self._timestamps.popleft()

            while len(self._timestamps) >= self._limit:
                sleep_for = self._timestamps[0] + self._interval - now
                if sleep_for <= 0:
                    self._timestamps.popleft()
                    now = loop.time()
                    continue
                await asyncio.sleep(sleep_for)
                now = loop.time()
                while self._timestamps and now - self._timestamps[0] >= self._interval:
                    self._timestamps.popleft()

            self._timestamps.append(loop.time())


async def _fetch_imdb(
    client: httpx.AsyncClient,
    imdb_id: str,
    *,
    cache: IMDbCache | None,
    throttle: _RequestThrottler | None,
    max_retries: int,
    backoff: float,
    retry_queue: IMDbRetryQueue | None,
) -> IMDbTitle | None:
    """Fetch a single IMDb title with retry, caching, and throttling."""

    if cache:
        cached = cache.get(imdb_id)
        if cached:
            return IMDbTitle.model_validate(cached)

    url = f"https://api.imdbapi.dev/titles/{imdb_id}"
    delay = backoff
    for attempt in range(max_retries + 1):
        if throttle is not None:
            await throttle.acquire()
        try:
            response = await client.get(url)
        except httpx.HTTPError:
            LOGGER.exception("HTTP error fetching IMDb ID %s", imdb_id)
            return None

        if response.status_code == 429:
            if attempt == max_retries:
                if retry_queue is not None:
                    await retry_queue.put(imdb_id)
                return None
            await asyncio.sleep(delay)
            delay *= 2
            continue

        if response.is_success:
            data = response.json()
            if cache:
                cache.set(imdb_id, data)
            return IMDbTitle.model_validate(data)

        return None

    return None


async def _fetch_imdb_batch(
    client: httpx.AsyncClient,
    imdb_ids: Sequence[str],
    *,
    cache: IMDbCache | None,
    throttle: _RequestThrottler | None,
    max_retries: int,
    backoff: float,
    retry_queue: IMDbRetryQueue | None,
    batch_limit: int,
) -> dict[str, IMDbTitle | None]:
    """Fetch metadata for multiple IMDb IDs using batch requests."""

    results: dict[str, IMDbTitle | None] = {}
    ids_to_fetch: list[str] = []
    for imdb_id in imdb_ids:
        if cache:
            cached = cache.get(imdb_id)
            if cached:
                results[imdb_id] = IMDbTitle.model_validate(cached)
                continue
        ids_to_fetch.append(imdb_id)

    if not ids_to_fetch:
        return results

    url = "https://api.imdbapi.dev/titles:batchGet"
    batch_size = require_positive(int(batch_limit), name="batch_limit")

    for start in range(0, len(ids_to_fetch), batch_size):
        chunk = ids_to_fetch[start : start + batch_size]
        params = [("titleIds", imdb_id) for imdb_id in chunk]
        delay = backoff
        for attempt in range(max_retries + 1):
            if throttle is not None:
                await throttle.acquire()
            try:
                response = await client.get(url, params=params)
            except httpx.HTTPError:
                LOGGER.exception(
                    "HTTP error fetching IMDb IDs %s", ",".join(chunk)
                )
                for imdb_id in chunk:
                    results[imdb_id] = None
                break

            if response.status_code == 429:
                if attempt == max_retries:
                    if retry_queue is not None:
                        for imdb_id in chunk:
                            await retry_queue.put(imdb_id)
                    for imdb_id in chunk:
                        results[imdb_id] = None
                    break
                await asyncio.sleep(delay)
                delay *= 2
                continue

            if response.is_success:
                data = response.json()
                found: set[str] = set()
                for title_data in data.get("titles", []):
                    imdb_title = IMDbTitle.model_validate(title_data)
                    results[imdb_title.id] = imdb_title
                    found.add(imdb_title.id)
                    if cache:
                        cache.set(imdb_title.id, title_data)
                for missing in set(chunk) - found:
                    results[missing] = None
                break

            for imdb_id in chunk:
                results[imdb_id] = None
            break

    return results
