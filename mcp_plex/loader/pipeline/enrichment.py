"""Enrichment stage coordinator for the loader pipeline.

Movie metadata enrichment has been ported from the legacy loader and now
performs TMDb and IMDb lookups before emitting aggregated payloads to the
persistence queue.  Episode enrichment reuses the TMDb caching and lookup
logic, and sample-mode batches pass straight through to persistence so end to
end processing mirrors the legacy worker implementation.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    asynccontextmanager,
)
from typing import Protocol, cast

import httpx
from pydantic import ValidationError

from .channels import (
    EpisodeBatch,
    IMDbRetryQueue,
    INGEST_DONE,
    PERSIST_DONE,
    IngestQueue,
    MovieBatch,
    PersistenceQueue,
    SampleBatch,
    chunk_sequence,
    enqueue_nowait,
)
from ...common.validation import coerce_plex_tag_id, require_positive

from ...common.types import (
    AggregatedItem,
    ExternalIDs,
    IMDbTitle,
    JSONValue,
    PlexGuid,
    PlexItem,
    PlexPerson,
    TMDBEpisode,
    TMDBItem,
    TMDBMovie,
    TMDBShow,
)
from ..imdb_cache import IMDbCache
from ..qdrant import rehydrate_aggregated_item

from plexapi.base import PlexPartialObject


LOGGER = logging.getLogger(__name__)


class AsyncHTTPClient(Protocol):
    """Minimal async HTTP client interface used by the enrichment stage."""

    async def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response: ...

    async def aclose(self) -> None: ...


HTTPClientResource = (
    AsyncHTTPClient
    | AbstractAsyncContextManager[AsyncHTTPClient]
    | AbstractContextManager[AsyncHTTPClient]
)

HTTPClientFactory = Callable[[], HTTPClientResource | Awaitable[HTTPClientResource]]
ExistingPayloadLookup = Callable[
    [ExternalIDs, str | None], Awaitable[Mapping[str, JSONValue] | None]
]


def _extract_external_ids(item: PlexPartialObject) -> ExternalIDs:
    """Extract IMDb and TMDb IDs from a Plex object."""

    imdb_id: str | None = None
    tmdb_id: str | None = None
    for guid in getattr(item, "guids", []) or []:
        gid = getattr(guid, "id", "")
        if gid.startswith("imdb://"):
            imdb_id = gid.split("imdb://", 1)[1]
        elif gid.startswith("tmdb://"):
            tmdb_id = gid.split("tmdb://", 1)[1]
    return ExternalIDs(imdb=imdb_id, tmdb=tmdb_id)


def _build_plex_item(item: PlexPartialObject) -> PlexItem:
    """Convert a Plex object into the internal :class:`PlexItem`."""

    guids = [PlexGuid(id=g.id) for g in getattr(item, "guids", [])]
    directors = [
        PlexPerson(
            id=coerce_plex_tag_id(getattr(d, "id", 0)),
            tag=str(getattr(d, "tag", "")),
            thumb=getattr(d, "thumb", None),
        )
        for d in getattr(item, "directors", []) or []
    ]
    writers = [
        PlexPerson(
            id=coerce_plex_tag_id(getattr(w, "id", 0)),
            tag=str(getattr(w, "tag", "")),
            thumb=getattr(w, "thumb", None),
        )
        for w in getattr(item, "writers", []) or []
    ]
    actors = [
        PlexPerson(
            id=coerce_plex_tag_id(getattr(a, "id", 0)),
            tag=str(getattr(a, "tag", "")),
            thumb=getattr(a, "thumb", None),
            role=getattr(a, "role", None),
        )
        for a in getattr(item, "actors", []) or getattr(item, "roles", []) or []
    ]
    genres = [
        str(getattr(g, "tag", ""))
        for g in getattr(item, "genres", []) or []
        if getattr(g, "tag", None)
    ]
    collections = [
        str(getattr(c, "tag", ""))
        for c in getattr(item, "collections", []) or []
        if getattr(c, "tag", None)
    ]
    season_number = getattr(item, "parentIndex", None)
    if isinstance(season_number, str):
        season_number = int(season_number) if season_number.isdigit() else None
    episode_number = getattr(item, "index", None)
    if isinstance(episode_number, str):
        episode_number = int(episode_number) if episode_number.isdigit() else None

    return PlexItem(
        rating_key=str(getattr(item, "ratingKey", "")),
        guid=str(getattr(item, "guid", "")),
        type=str(getattr(item, "type", "")), # type: ignore
        title=str(getattr(item, "title", "")),
        show_title=getattr(item, "grandparentTitle", None),
        season_title=getattr(item, "parentTitle", None),
        season_number=season_number,
        episode_number=episode_number,
        summary=getattr(item, "summary", None),
        year=getattr(item, "year", None),
        added_at=getattr(item, "addedAt", None),
        guids=guids,
        thumb=getattr(item, "thumb", None),
        art=getattr(item, "art", None),
        tagline=getattr(item, "tagline", None),
        content_rating=getattr(item, "contentRating", None),
        directors=directors,
        writers=writers,
        actors=actors,
        genres=genres,
        collections=collections,
    )


async def _fetch_tmdb_movie(
    client: AsyncHTTPClient, tmdb_id: str, api_key: str
) -> TMDBMovie | None:
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=reviews"
    try:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError:
        LOGGER.exception("HTTP error fetching TMDb movie %s", tmdb_id)
        return None
    if resp.is_success:
        return TMDBMovie.model_validate(resp.json())
    return None


async def _fetch_tmdb_show(
    client: AsyncHTTPClient, tmdb_id: str, api_key: str
) -> TMDBShow | None:
    url = f"https://api.themoviedb.org/3/tv/{tmdb_id}?append_to_response=reviews"
    try:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError:
        LOGGER.exception("HTTP error fetching TMDb show %s", tmdb_id)
        return None
    if resp.is_success:
        return TMDBShow.model_validate(resp.json())
    return None


async def _fetch_tmdb_episode(
    client: AsyncHTTPClient,
    show_id: int,
    season_number: int,
    episode_number: int,
    api_key: str,
) -> TMDBEpisode | None:
    """Fetch TMDb data for a TV episode."""

    url = f"https://api.themoviedb.org/3/tv/{show_id}/season/{season_number}/episode/{episode_number}"
    try:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError:
        LOGGER.exception(
            "HTTP error fetching TMDb episode %s S%sE%s",
            show_id,
            season_number,
            episode_number,
        )
        return None
    if resp.is_success:
        data = resp.json()
        if isinstance(data, dict):
            data.setdefault("show_id", show_id)
        return TMDBEpisode.model_validate(data)
    return None


async def _fetch_tmdb_episode_chunk(
    client: AsyncHTTPClient,
    show_id: int,
    append_paths: Sequence[str],
    api_key: str,
) -> dict[str, TMDBEpisode | None]:
    """Fetch multiple TMDb episodes for *show_id* in a single request."""

    if not append_paths:
        return {}
    url = f"https://api.themoviedb.org/3/tv/{show_id}"
    params = {"append_to_response": ",".join(append_paths)}
    try:
        resp = await client.get(
            url, params=params, headers={"Authorization": f"Bearer {api_key}"}
        )
    except httpx.HTTPError:
        LOGGER.exception("HTTP error fetching TMDb episode chunk for show %s", show_id)
        return {}
    if not resp.is_success:
        return {}

    data = resp.json()
    results: dict[str, TMDBEpisode | None] = {}
    for path in append_paths:
        payload = data.get(path)
        if isinstance(payload, dict):
            payload.setdefault("show_id", show_id)
            try:
                results[path] = TMDBEpisode.model_validate(payload)
            except ValidationError:
                LOGGER.exception(
                    "Validation error parsing TMDb episode payload for %s", path
                )
                results[path] = None
        else:
            results[path] = None
    return results


def resolve_tmdb_season_number(
    show_tmdb: TMDBShow | None, episode: PlexPartialObject
) -> int | None:
    """Map a Plex episode to the appropriate TMDb season number.

    This resolves cases where Plex uses year-based season indices that do not
    match TMDb's sequential ``season_number`` values.
    """

    parent_index = getattr(episode, "parentIndex", None)
    parent_title = getattr(episode, "parentTitle", None)
    parent_year = getattr(episode, "parentYear", None)
    if parent_year is None:
        parent_year = getattr(episode, "year", None)

    seasons = getattr(show_tmdb, "seasons", []) if show_tmdb else []

    # direct numeric match
    if parent_index is not None:
        for season in seasons:
            if season.season_number == parent_index:
                return season.season_number

    # match by season name (e.g. "Season 2018" -> "2018")
    title_norm: str | None = None
    if isinstance(parent_title, str):
        title_norm = parent_title.lower().lstrip("season ").strip()
        for season in seasons:
            name_norm = (season.name or "").lower().lstrip("season ").strip()
            if name_norm == title_norm:
                return season.season_number

    # match by air date year when Plex uses year-based seasons
    year: int | None = None
    if isinstance(parent_year, int):
        year = parent_year
    elif isinstance(parent_index, int):
        year = parent_index
    elif title_norm and title_norm.isdigit():
        year = int(title_norm)

    if year is not None:
        for season in seasons:
            air = getattr(season, "air_date", None)
            if isinstance(air, str) and len(air) >= 4 and air[:4].isdigit():
                if int(air[:4]) == year:
                    return season.season_number

    if isinstance(parent_index, int):
        return parent_index
    if isinstance(parent_index, str) and parent_index.isdigit():
        return int(parent_index)
    if isinstance(parent_title, str) and parent_title.isdigit():
        return int(parent_title)
    return None


class EnrichmentStage:
    """Coordinate metadata enrichment for ingested media batches."""

    def __init__(
        self,
        *,
        http_client_factory: HTTPClientFactory,
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
        idle_retry_delay: float = 0.05,
        existing_payload_lookup: ExistingPayloadLookup | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._http_client_factory: HTTPClientFactory = http_client_factory
        self._tmdb_api_key = (tmdb_api_key or "").strip()
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
            raise ValueError("imdb_requests_per_window must be positive when provided")
        if imdb_window_seconds <= 0:
            raise ValueError("imdb_window_seconds must be positive")
        self._imdb_throttle = _RequestThrottler(
            limit=imdb_requests_per_window,
            interval=float(imdb_window_seconds),
        )
        if idle_retry_delay < 0:
            raise ValueError("idle_retry_delay must be non-negative")
        self._idle_retry_delay = float(idle_retry_delay)
        self._existing_payload_lookup = existing_payload_lookup
        requested_movie_batch_size = require_positive(
            int(movie_batch_size), name="movie_batch_size"
        )
        self._movie_batch_size = min(requested_movie_batch_size, 100)
        self._episode_batch_size = require_positive(
            int(episode_batch_size), name="episode_batch_size"
        )
        self._logger = logger or logging.getLogger("mcp_plex.loader.enrichment")
        self._show_tmdb_cache: dict[str, TMDBShow | None] = {}
        self._pending_imdb_items: dict[str, list[AggregatedItem]] = {}

    @property
    def logger(self) -> logging.Logger:
        """Logger used by the enrichment stage."""

        return self._logger

    @property
    def imdb_retry_queue(self) -> IMDbRetryQueue:
        """IMDb retry queue used by the enrichment stage."""

        return self._imdb_retry_queue

    @property
    def idle_retry_delay(self) -> float:
        """Seconds waited after idle retry cycles to yield the event loop."""

        return self._idle_retry_delay

    async def run(self) -> None:
        """Execute the enrichment stage."""

        self._logger.info(
            "Starting enrichment stage with movie batch size=%d and episode batch size=%d.",
            self._movie_batch_size,
            self._episode_batch_size,
        )
        while True:
            got_item = False
            try:
                batch = self._ingest_queue.get_nowait()
                got_item = True
            except asyncio.QueueEmpty:
                if await self._retry_imdb_batches():
                    await self._idle_pause()
                    continue
                batch = await self._ingest_queue.get()
                got_item = True
            try:
                if batch is None:
                    self._logger.debug("Received legacy completion token; ignoring.")
                    continue

                if batch is INGEST_DONE:
                    self._logger.info(
                        "Ingestion completed; finishing enrichment stage."
                    )
                    break

                if isinstance(batch, MovieBatch):
                    self._logger.info(
                        "Enriching movie batch with %d item(s) (ingest queue=%d).",
                        len(batch.movies),
                        self._ingest_queue.qsize(),
                    )
                    await self._handle_movie_batch(batch)
                elif isinstance(batch, EpisodeBatch):
                    self._logger.info(
                        "Enriching episode batch for %s with %d item(s) (ingest queue=%d).",
                        getattr(batch.show, "title", str(batch.show)),
                        len(batch.episodes),
                        self._ingest_queue.qsize(),
                    )
                    await self._handle_episode_batch(batch)
                elif isinstance(batch, SampleBatch):
                    self._logger.info(
                        "Forwarding sample batch with %d item(s) (ingest queue=%d).",
                        len(batch.items),
                        self._ingest_queue.qsize(),
                    )
                    await self._handle_sample_batch(batch)
                else:  # pragma: no cover - defensive logging for future types
                    self._logger.warning("Received unsupported batch type: %r", batch)
            finally:
                if got_item:
                    self._ingest_queue.task_done()

        await enqueue_nowait(self._persistence_queue, PERSIST_DONE)
        self._logger.info(
            "Enrichment stage completed; persistence sentinel emitted (retry queue=%d).",
            self._imdb_retry_queue.qsize(),
        )

    async def _idle_pause(self) -> None:
        """Yield control after retry work to avoid busy-looping when idle."""

        delay = self._idle_retry_delay
        if delay <= 0:
            await asyncio.sleep(0)
            return
        await asyncio.sleep(delay)

    async def _handle_movie_batch(self, batch: MovieBatch) -> None:
        """Enrich and forward Plex movie batches to the persistence stage."""

        movie_chunks = [
            chunk
            for chunk in chunk_sequence(batch.movies, self._movie_batch_size)
            if chunk
        ]
        if not movie_chunks:
            return

        async with self._acquire_http_client() as client:
            for movies in movie_chunks:
                reused_items, remaining_movies = await self._split_reused_items(movies)
                if reused_items:
                    await self._emit_persistence_batch(reused_items)
                if not remaining_movies:
                    continue
                aggregated = await self._enrich_movies(client, remaining_movies)
                await self._emit_persistence_batch(aggregated)
                self._logger.info(
                    "Processed movie batch with %d items (queue size=%d)",
                    len(aggregated),
                    self._persistence_queue.qsize(),
                )

    async def _handle_episode_batch(self, batch: EpisodeBatch) -> None:
        """Enrich and forward Plex episode batches to the persistence stage."""

        episode_chunks = [
            chunk
            for chunk in chunk_sequence(batch.episodes, self._episode_batch_size)
            if chunk
        ]
        if not episode_chunks:
            return

        show_title = getattr(batch.show, "title", str(batch.show))
        async with self._acquire_http_client() as client:
            for episodes in episode_chunks:
                reused_items, remaining_episodes = await self._split_reused_items(
                    episodes
                )
                if reused_items:
                    await self._emit_persistence_batch(reused_items)
                if not remaining_episodes:
                    continue
                aggregated = await self._enrich_episodes(
                    client, batch.show, remaining_episodes
                )
                await self._emit_persistence_batch(aggregated)
                self._logger.info(
                    "Processed episode batch for %s with %d items (queue size=%d)",
                    show_title,
                    len(aggregated),
                    self._persistence_queue.qsize(),
                )

    @asynccontextmanager
    async def _acquire_http_client(self) -> AsyncIterator[AsyncHTTPClient]:
        """Yield an HTTP client from the injected factory."""

        resource = self._http_client_factory()
        if inspect.isawaitable(resource):
            resource = await resource

        if isinstance(resource, AbstractAsyncContextManager):
            async with resource as client:
                yield client
            return

        if isinstance(resource, AbstractContextManager):
            with resource as client:
                yield client
            return

        if hasattr(resource, "__aenter__") and hasattr(resource, "__aexit__"):
            async with cast(
                AbstractAsyncContextManager[AsyncHTTPClient], resource
            ) as client:
                yield client
            return

        if hasattr(resource, "__enter__") and hasattr(resource, "__exit__"):
            with cast(AbstractContextManager[AsyncHTTPClient], resource) as client:
                yield client
            return

        client = cast(AsyncHTTPClient, resource)
        try:
            yield client
        finally:
            try:
                await client.aclose()
            except AttributeError:
                closer = getattr(client, "close", None)
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
        payload = list(aggregated)
        await enqueue_nowait(self._persistence_queue, payload)
        self._logger.debug(
            "Enqueued %d aggregated item(s) for persistence (queue size=%d).",
            len(payload),
            self._persistence_queue.qsize(),
        )

    async def _split_reused_items(
        self, items: Sequence[PlexPartialObject]
    ) -> tuple[list[AggregatedItem], list[PlexPartialObject]]:
        """Return reused payloads and remaining Plex items to enrich."""

        if not items:
            return [], []

        lookup = self._existing_payload_lookup
        if lookup is None:
            return [], list(items)

        reused: list[AggregatedItem] = []
        remaining: list[PlexPartialObject] = []
        for item in items:
            reused_item = await self._reuse_payload_if_mismatch(lookup, item)
            if reused_item is None:
                remaining.append(item)
            else:
                reused.append(reused_item)
        return reused, remaining

    async def _reuse_payload_if_mismatch(
        self,
        lookup: ExistingPayloadLookup,
        item: PlexPartialObject,
    ) -> AggregatedItem | None:
        external_ids = _extract_external_ids(item)
        plex_guid = str(getattr(item, "guid", "")) or None
        if not (external_ids.imdb or external_ids.tmdb or plex_guid):
            return None

        payload = await lookup(external_ids, plex_guid)
        if payload is None:
            return None

        try:
            aggregated = rehydrate_aggregated_item(payload)
        except (ValueError, ValidationError) as exc:
            self._logger.debug(
                "Failed to rehydrate Qdrant payload for Plex item %s.",
                getattr(item, "ratingKey", ""),
                exc_info=exc,
            )
            return None

        current_rating_key = str(getattr(item, "ratingKey", ""))
        if not current_rating_key:
            return None
        if aggregated.plex.rating_key == current_rating_key:
            return None

        updated_plex = aggregated.plex.model_copy(
            update={"rating_key": current_rating_key}
        )
        self._logger.info(
            "Detected rating key mismatch for %s; reusing existing payload.",
            current_rating_key,
        )
        return aggregated.model_copy(update={"plex": updated_plex})

    async def _enrich_movies(
        self, client: AsyncHTTPClient, movies: Sequence[PlexPartialObject]
    ) -> list[AggregatedItem]:
        """Fetch external metadata for *movies* and aggregate the results."""

        movie_ids = [_extract_external_ids(movie) for movie in movies]
        imdb_ids = [ids.imdb for ids in movie_ids if ids.imdb]
        imdb_future: asyncio.Task[dict[str, IMDbTitle | None]] | None = None
        if imdb_ids:
            imdb_future = asyncio.create_task(
                _fetch_imdb_batch(
                    client,
                    imdb_ids,
                    cache=self._imdb_cache,
                    throttle=self._imdb_throttle,
                    max_retries=self._imdb_max_retries,
                    backoff=self._imdb_backoff,
                    retry_queue=self._imdb_retry_queue,
                    batch_limit=self._imdb_batch_limit,
                )
            )

        api_key = self._tmdb_api_key
        tmdb_tasks: list[asyncio.Task[TMDBMovie | None]] = []
        if api_key:
            for ids in movie_ids:
                if not ids.tmdb:
                    continue
                tmdb_tasks.append(
                    asyncio.create_task(_fetch_tmdb_movie(client, ids.tmdb, api_key))
                )

        imdb_map: dict[str, IMDbTitle | None] = {}
        retry_snapshot: set[str] = set()
        tmdb_results: list[TMDBMovie | None] = []
        if imdb_future is not None:
            combined_results = await asyncio.gather(imdb_future, *tmdb_tasks)
            imdb_map = cast(dict[str, IMDbTitle | None], combined_results[0])
            tmdb_results = [
                cast(TMDBMovie | None, result) for result in combined_results[1:]
            ]
            retry_snapshot = set(self._imdb_retry_queue.snapshot())
        elif tmdb_tasks:
            tmdb_results = [await task for task in tmdb_tasks]

        tmdb_iter = iter(tmdb_results)

        aggregated: list[AggregatedItem] = []
        for movie, ids in zip(movies, movie_ids):
            tmdb = next(tmdb_iter, None) if ids.tmdb else None
            imdb = imdb_map.get(ids.imdb) if ids.imdb else None
            aggregated_item = AggregatedItem(
                plex=_build_plex_item(movie),
                imdb=imdb,
                tmdb=tmdb,
            )
            aggregated.append(aggregated_item)
            if ids.imdb and imdb is None and ids.imdb in retry_snapshot:
                self._register_pending_imdb(ids.imdb, aggregated_item)
        return aggregated

    async def _handle_sample_batch(self, batch: SampleBatch) -> None:
        """Forward sample data batches directly to the persistence stage."""

        for aggregated in chunk_sequence(batch.items, self._movie_batch_size):
            if not aggregated:
                continue
            await self._emit_persistence_batch(aggregated)
            self._logger.info(
                "Processed sample batch with %d items (queue size=%d)",
                len(aggregated),
                self._persistence_queue.qsize(),
            )

    async def _enrich_episodes(
        self,
        client: AsyncHTTPClient,
        show: PlexPartialObject,
        episodes: Sequence[PlexPartialObject],
    ) -> list[AggregatedItem]:
        """Fetch external metadata for *episodes* and aggregate the results."""

        show_ids = _extract_external_ids(show)
        imdb_future: asyncio.Task[dict[str, IMDbTitle | None]] | None = None
        show_future: asyncio.Task[TMDBShow | None] | None = None
        show_tmdb: TMDBShow | None = None
        if show_ids.tmdb:
            show_future = asyncio.create_task(
                self._get_tmdb_show(client, show_ids.tmdb)
            )
        episode_ids = [_extract_external_ids(ep) for ep in episodes]
        imdb_ids = [ids.imdb for ids in episode_ids if ids.imdb]
        if imdb_ids:
            imdb_future = asyncio.create_task(
                _fetch_imdb_batch(
                    client,
                    imdb_ids,
                    cache=self._imdb_cache,
                    throttle=self._imdb_throttle,
                    max_retries=self._imdb_max_retries,
                    backoff=self._imdb_backoff,
                    retry_queue=self._imdb_retry_queue,
                    batch_limit=self._imdb_batch_limit,
                )
            )

        if show_future is not None:
            show_tmdb = await show_future

        tmdb_future: asyncio.Task[list[TMDBEpisode | None]] | None = None
        if show_tmdb:
            tmdb_future = asyncio.create_task(
                self._bulk_lookup_tmdb_episodes(client, show_tmdb, episodes)
            )

        imdb_map: dict[str, IMDbTitle | None] = {}
        retry_snapshot: set[str] = set()
        tmdb_results: list[TMDBEpisode | None] = [None] * len(episodes)
        if imdb_future and tmdb_future:
            imdb_map, tmdb_results = await asyncio.gather(imdb_future, tmdb_future)
            retry_snapshot = set(self._imdb_retry_queue.snapshot())
        elif imdb_future:
            imdb_map = await imdb_future
            retry_snapshot = set(self._imdb_retry_queue.snapshot())
        elif tmdb_future:
            tmdb_results = await tmdb_future

        tmdb_iter = iter(tmdb_results)

        aggregated: list[AggregatedItem] = []
        for ep, ids in zip(episodes, episode_ids):
            tmdb_episode = next(tmdb_iter, None)
            imdb = imdb_map.get(ids.imdb) if ids.imdb else None
            tmdb_item: TMDBItem | None = tmdb_episode or show_tmdb
            aggregated_item = AggregatedItem(
                plex=_build_plex_item(ep),
                imdb=imdb,
                tmdb=tmdb_item,
            )
            aggregated.append(aggregated_item)
            if ids.imdb and imdb is None and ids.imdb in retry_snapshot:
                self._register_pending_imdb(ids.imdb, aggregated_item)
        return aggregated

    async def _get_tmdb_show(
        self, client: AsyncHTTPClient, tmdb_id: str
    ) -> TMDBShow | None:
        """Return the TMDb show for *tmdb_id*, using the in-memory cache."""

        if not self._tmdb_api_key:
            return None
        tmdb_id_str = str(tmdb_id)
        if tmdb_id_str in self._show_tmdb_cache:
            return self._show_tmdb_cache[tmdb_id_str]
        show = await _fetch_tmdb_show(client, tmdb_id_str, self._tmdb_api_key or "")
        self._show_tmdb_cache[tmdb_id_str] = show
        return show

    async def _bulk_lookup_tmdb_episodes(
        self,
        client: AsyncHTTPClient,
        show_tmdb: TMDBShow,
        episodes: Sequence[PlexPartialObject],
    ) -> list[TMDBEpisode | None]:
        """Fetch TMDb metadata for *episodes* in batches."""

        results: list[TMDBEpisode | None] = [None] * len(episodes)
        if not self._tmdb_api_key:
            return results

        lookups: list[tuple[int, int, int]] = []
        for idx, episode in enumerate(episodes):
            season = resolve_tmdb_season_number(show_tmdb, episode)
            ep_num = getattr(episode, "index", None)
            if isinstance(ep_num, str) and ep_num.isdigit():
                ep_num = int(ep_num)
            if season is None or ep_num is None:
                continue
            lookups.append((idx, int(season), int(ep_num)))

        if not lookups:
            return results

        for chunk in chunk_sequence(lookups, 20):
            chunk_map: dict[str, list[int]] = {}
            for idx, season, ep_num in chunk:
                path = f"season/{season}/episode/{ep_num}"
                chunk_map.setdefault(path, []).append(idx)
            chunk_paths = list(chunk_map.keys())
            chunk_results = await _fetch_tmdb_episode_chunk(
                client, show_tmdb.id, chunk_paths, self._tmdb_api_key
            )
            for path, indexes in chunk_map.items():
                episode_result = chunk_results.get(path)
                if episode_result is None:
                    continue
                for idx in indexes:
                    results[idx] = episode_result

        for idx, season, ep_num in lookups:
            if results[idx] is not None:
                continue
            results[idx] = await _fetch_tmdb_episode(
                client,
                show_tmdb.id,
                season,
                ep_num,
                self._tmdb_api_key,
            )

        return results

    async def _retry_imdb_batches(self) -> bool:
        """Process IMDb IDs from the retry queue when idle."""

        if self._imdb_retry_queue.empty():
            return False

        self._logger.debug(
            "Processing IMDb retry queue with %d pending id(s).",
            self._imdb_retry_queue.qsize(),
        )
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
            imdb_results = await _fetch_imdb_batch(
                client,
                imdb_ids,
                cache=self._imdb_cache,
                throttle=self._imdb_throttle,
                max_retries=self._imdb_max_retries,
                backoff=self._imdb_backoff,
                retry_queue=self._imdb_retry_queue,
                batch_limit=self._imdb_batch_limit,
            )

        updated: list[AggregatedItem] = []
        for imdb_id in imdb_ids:
            imdb_title = imdb_results.get(imdb_id)
            if imdb_title is None:
                continue
            pending_items = self._pending_imdb_items.pop(imdb_id, [])
            for item in pending_items:
                updated.append(item.model_copy(update={"imdb": imdb_title}))

        if updated:
            await self._emit_persistence_batch(updated)

        stalled_ids = [
            imdb_id for imdb_id in imdb_ids if imdb_results.get(imdb_id) is None
        ]
        for imdb_id in stalled_ids:
            self._imdb_retry_queue.put_nowait(imdb_id)

        self._logger.info(
            "Retried IMDb batch with %d updates (retry queue=%d)",
            len(updated),
            self._imdb_retry_queue.qsize(),
        )
        progress_made = bool(updated) or len(stalled_ids) < len(imdb_ids)
        return progress_made

    def _register_pending_imdb(self, imdb_id: str, item: AggregatedItem) -> None:
        """Track *item* for re-emission once IMDb metadata becomes available."""

        items = self._pending_imdb_items.setdefault(imdb_id, [])
        items.append(item)


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
    client: AsyncHTTPClient,
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
                    retry_queue.put_nowait(imdb_id)
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
    client: AsyncHTTPClient,
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
                response = await client.get(url, params=params) # type: ignore
            except httpx.HTTPError:
                LOGGER.exception("HTTP error fetching IMDb IDs %s", ",".join(chunk))
                for imdb_id in chunk:
                    results[imdb_id] = None
                break

            if response.status_code == 429:
                if attempt == max_retries:
                    if retry_queue is not None:
                        for imdb_id in chunk:
                            retry_queue.put_nowait(imdb_id)
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
