"""Utilities for loading Plex metadata with IMDb and TMDb details."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Awaitable, Iterable, List, Optional, Sequence, TypeVar

import click
import httpx
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from .imdb_cache import IMDbCache
from .types import (
    AggregatedItem,
    ExternalIDs,
    IMDbTitle,
    PlexGuid,
    PlexItem,
    PlexPerson,
    TMDBEpisode,
    TMDBItem,
    TMDBMovie,
    TMDBShow,
)

try:  # Only import plexapi when available; the sample data mode does not require it.
    from plexapi.base import PlexPartialObject
    from plexapi.server import PlexServer
except Exception:
    PlexServer = None  # type: ignore[assignment]
    PlexPartialObject = object  # type: ignore[assignment]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

T = TypeVar("T")

_imdb_cache: IMDbCache | None = None
_imdb_max_retries: int = 3
_imdb_backoff: float = 1.0
_imdb_retry_queue: "_IMDbRetryQueue" | None = None
_imdb_batch_limit: int = 5
_qdrant_batch_size: int = 1000
_qdrant_upsert_buffer_size: int = 200
_qdrant_max_concurrent_upserts: int = 4


def _require_positive(value: int, *, name: str) -> int:
    """Return *value* if positive, otherwise raise a ``ValueError``."""

    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class _IMDbRetryQueue(asyncio.Queue[str]):
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
            raise RuntimeError("Desynchronization: Queue is not empty but self._items is empty.")
        try:
            item = super().get_nowait()
        except asyncio.QueueEmpty:
            raise RuntimeError("Desynchronization: self._items is not empty but asyncio.Queue is empty.")
        self._items.popleft()
        return item

    def snapshot(self) -> list[str]:
        """Return a list of the current queue contents."""

        return list(self._items)

# Known Qdrant-managed dense embedding models with their dimensionality and
# similarity metric. To support a new server-side embedding model, add an entry
# here with the appropriate vector size and `models.Distance` value.
_DENSE_MODEL_PARAMS: dict[str, tuple[int, models.Distance]] = {
    "BAAI/bge-small-en-v1.5": (384, models.Distance.COSINE),
    "BAAI/bge-base-en-v1.5": (768, models.Distance.COSINE),
    "BAAI/bge-large-en-v1.5": (1024, models.Distance.COSINE),
    "text-embedding-3-small": (1536, models.Distance.COSINE),
    "text-embedding-3-large": (3072, models.Distance.COSINE),
}


async def _iter_gather_in_batches(
    tasks: Sequence[Awaitable[T]], batch_size: int
) -> AsyncIterator[T]:
    """Yield results from awaitable tasks in fixed-size batches."""

    _require_positive(batch_size, name="batch_size")

    total = len(tasks)
    for i in range(0, total, batch_size):
        batch = tasks[i : i + batch_size]
        for result in await asyncio.gather(*batch):
            yield result
        logger.info("Processed %d/%d items", min(i + batch_size, total), total)


async def _gather_in_batches(
    tasks: Sequence[Awaitable[T]], batch_size: int
) -> List[T]:
    """Gather awaitable tasks in fixed-size batches."""

    return [result async for result in _iter_gather_in_batches(tasks, batch_size)]


def _resolve_dense_model_params(model_name: str) -> tuple[int, models.Distance]:
    """Look up Qdrant vector parameters for a known dense embedding model."""

    try:
        return _DENSE_MODEL_PARAMS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dense embedding model '{model_name}'. Update _DENSE_MODEL_PARAMS with the model's size and distance."
        ) from exc


async def _fetch_imdb(client: httpx.AsyncClient, imdb_id: str) -> Optional[IMDbTitle]:
    """Fetch metadata for an IMDb ID with caching and retry logic."""

    if _imdb_cache:
        cached = _imdb_cache.get(imdb_id)
        if cached:
            return IMDbTitle.model_validate(cached)

    url = f"https://api.imdbapi.dev/titles/{imdb_id}"
    delay = _imdb_backoff
    for attempt in range(_imdb_max_retries + 1):
        try:
            resp = await client.get(url)
        except httpx.HTTPError:
            logger.exception("HTTP error fetching IMDb ID %s", imdb_id)
            return None
        if resp.status_code == 429:
            if attempt == _imdb_max_retries:
                if _imdb_retry_queue is not None:
                    await _imdb_retry_queue.put(imdb_id)
                return None
            await asyncio.sleep(delay)
            delay *= 2
            continue
        if resp.is_success:
            data = resp.json()
            if _imdb_cache:
                _imdb_cache.set(imdb_id, data)
            return IMDbTitle.model_validate(data)
        return None
    return None


async def _fetch_imdb_batch(
    client: httpx.AsyncClient, imdb_ids: Sequence[str]
) -> dict[str, Optional[IMDbTitle]]:
    """Fetch metadata for multiple IMDb IDs, batching requests."""

    results: dict[str, Optional[IMDbTitle]] = {}
    ids_to_fetch: list[str] = []
    for imdb_id in imdb_ids:
        if _imdb_cache:
            cached = _imdb_cache.get(imdb_id)
            if cached:
                results[imdb_id] = IMDbTitle.model_validate(cached)
                continue
        ids_to_fetch.append(imdb_id)

    if not ids_to_fetch:
        return results

    url = "https://api.imdbapi.dev/titles:batchGet"
    for i in range(0, len(ids_to_fetch), _imdb_batch_limit):
        chunk = ids_to_fetch[i : i + _imdb_batch_limit]
        params = [("titleIds", imdb_id) for imdb_id in chunk]
        delay = _imdb_backoff
        for attempt in range(_imdb_max_retries + 1):
            try:
                resp = await client.get(url, params=params)
            except httpx.HTTPError:
                logger.exception("HTTP error fetching IMDb IDs %s", ",".join(chunk))
                for imdb_id in chunk:
                    results[imdb_id] = None
                break
            if resp.status_code == 429:
                if attempt == _imdb_max_retries:
                    if _imdb_retry_queue is not None:
                        for imdb_id in chunk:
                            await _imdb_retry_queue.put(imdb_id)
                    for imdb_id in chunk:
                        results[imdb_id] = None
                    break
                await asyncio.sleep(delay)
                delay *= 2
                continue
            if resp.is_success:
                data = resp.json()
                found: set[str] = set()
                for title_data in data.get("titles", []):
                    imdb_title = IMDbTitle.model_validate(title_data)
                    results[imdb_title.id] = imdb_title
                    found.add(imdb_title.id)
                    if _imdb_cache:
                        _imdb_cache.set(imdb_title.id, title_data)
                for missing in set(chunk) - found:
                    results[missing] = None
                break
            for imdb_id in chunk:
                results[imdb_id] = None
            break

    return results


def _load_imdb_retry_queue(path: Path) -> None:
    """Populate the retry queue from a JSON file if it exists."""

    global _imdb_retry_queue
    ids: list[str] = []
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                ids = [str(imdb_id) for imdb_id in data]
            else:
                logger.warning(
                    "IMDb retry queue file %s did not contain a list; ignoring its contents",
                    path,
                )
        except Exception:
            logger.exception("Failed to load IMDb retry queue from %s", path)
    _imdb_retry_queue = _IMDbRetryQueue(ids)


async def _process_imdb_retry_queue(client: httpx.AsyncClient) -> None:
    """Attempt to fetch queued IMDb IDs, re-queueing failures."""

    if _imdb_retry_queue is None or _imdb_retry_queue.empty():
        return
    size = _imdb_retry_queue.qsize()
    for _ in range(size):
        imdb_id = await _imdb_retry_queue.get()
        title = await _fetch_imdb(client, imdb_id)
        if title is None:
            await _imdb_retry_queue.put(imdb_id)


def _persist_imdb_retry_queue(path: Path) -> None:
    """Persist the retry queue to disk."""

    if _imdb_retry_queue is None:
        return
    path.write_text(json.dumps(_imdb_retry_queue.snapshot()))


async def _upsert_in_batches(
    client: AsyncQdrantClient,
    collection_name: str,
    points: Sequence[models.PointStruct],
) -> None:
    """Upsert points into Qdrant in batches, logging HTTP errors."""

    total = len(points)
    for i in range(0, total, _qdrant_batch_size):
        batch = points[i : i + _qdrant_batch_size]
        try:
            await client.upsert(collection_name=collection_name, points=batch)
        except Exception:
            logger.exception(
                "Failed to upsert batch %d-%d", i, i + len(batch)
            )
        else:
            logger.info(
                "Upserted %d/%d points", min(i + len(batch), total), total
            )


async def _ensure_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    dense_size: int,
    dense_distance: models.Distance,
) -> None:
    """Create the collection and payload indexes if they do not already exist."""

    created_collection = False
    if not await client.collection_exists(collection_name):
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": models.VectorParams(size=dense_size, distance=dense_distance)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        created_collection = True

    if not created_collection:
        return

    text_index = models.TextIndexParams(
        type=models.PayloadSchemaType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        lowercase=True,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="title",
        field_schema=text_index,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="type",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="year",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="added_at",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="actors",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="directors",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="writers",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="genres",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="show_title",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="season_number",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="episode_number",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="collections",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="summary",
        field_schema=text_index,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="overview",
        field_schema=text_index,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="plot",
        field_schema=text_index,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="tagline",
        field_schema=text_index,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="reviews",
        field_schema=text_index,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="data.plex.rating_key",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="data.imdb.id",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    await client.create_payload_index(
        collection_name=collection_name,
        field_name="data.tmdb.id",
        field_schema=models.PayloadSchemaType.INTEGER,
    )


async def _fetch_tmdb_movie(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBMovie]:
    url = (
        f"https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=reviews"
    )
    try:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError:
        logger.exception("HTTP error fetching TMDb movie %s", tmdb_id)
        return None
    if resp.is_success:
        return TMDBMovie.model_validate(resp.json())
    return None


async def _fetch_tmdb_show(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBShow]:
    url = (
        f"https://api.themoviedb.org/3/tv/{tmdb_id}?append_to_response=reviews"
    )
    try:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError:
        logger.exception("HTTP error fetching TMDb show %s", tmdb_id)
        return None
    if resp.is_success:
        return TMDBShow.model_validate(resp.json())
    return None


async def _fetch_tmdb_episode(
    client: httpx.AsyncClient,
    show_id: int,
    season_number: int,
    episode_number: int,
    api_key: str,
) -> Optional[TMDBEpisode]:
    """Fetch TMDb data for a TV episode."""

    url = (
        f"https://api.themoviedb.org/3/tv/{show_id}/season/{season_number}/episode/{episode_number}"
    )
    try:
        resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    except httpx.HTTPError:
        logger.exception(
            "HTTP error fetching TMDb episode %s S%sE%s",
            show_id,
            season_number,
            episode_number,
        )
        return None
    if resp.is_success:
        return TMDBEpisode.model_validate(resp.json())
    return None


def resolve_tmdb_season_number(
    show_tmdb: Optional[TMDBShow], episode: PlexPartialObject
) -> Optional[int]:
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
    title_norm: Optional[str] = None
    if isinstance(parent_title, str):
        title_norm = parent_title.lower().lstrip("season ").strip()
        for season in seasons:
            name_norm = (season.name or "").lower().lstrip("season ").strip()
            if name_norm == title_norm:
                return season.season_number

    # match by air date year when Plex uses year-based seasons
    year: Optional[int] = None
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


def _extract_external_ids(item: PlexPartialObject) -> ExternalIDs:
    """Extract IMDb and TMDb IDs from a Plex object."""

    imdb_id: Optional[str] = None
    tmdb_id: Optional[str] = None
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
        PlexPerson(id=getattr(d, "id", 0), tag=str(getattr(d, "tag", "")), thumb=getattr(d, "thumb", None))
        for d in getattr(item, "directors", []) or []
    ]
    writers = [
        PlexPerson(id=getattr(w, "id", 0), tag=str(getattr(w, "tag", "")), thumb=getattr(w, "thumb", None))
        for w in getattr(item, "writers", []) or []
    ]
    actors = [
        PlexPerson(
            id=getattr(a, "id", 0),
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
        type=str(getattr(item, "type", "")),
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


async def _iter_from_plex(
    server: PlexServer, tmdb_api_key: str, *, batch_size: int = 50
) -> AsyncIterator[AggregatedItem]:
    """Yield items from a live Plex server as they are enriched."""

    async with httpx.AsyncClient(timeout=30) as client:
        movie_section = server.library.section("Movies")
        movie_keys = [int(m.ratingKey) for m in movie_section.all()]
        movies = server.fetchItems(movie_keys) if movie_keys else []
        movie_imdb_ids = [
            _extract_external_ids(m).imdb for m in movies if _extract_external_ids(m).imdb
        ]
        movie_imdb_map = (
            await _fetch_imdb_batch(client, movie_imdb_ids) if movie_imdb_ids else {}
        )

        async def _augment_movie(movie: PlexPartialObject) -> AggregatedItem:
            ids = _extract_external_ids(movie)
            imdb = movie_imdb_map.get(ids.imdb) if ids.imdb else None
            tmdb = (
                await _fetch_tmdb_movie(client, ids.tmdb, tmdb_api_key)
                if ids.tmdb
                else None
            )
            return AggregatedItem(plex=_build_plex_item(movie), imdb=imdb, tmdb=tmdb)

        movie_tasks = [_augment_movie(movie) for movie in movies]
        if movie_tasks:
            async for result in _iter_gather_in_batches(movie_tasks, batch_size):
                yield result

        show_section = server.library.section("TV Shows")
        show_keys = [int(s.ratingKey) for s in show_section.all()]
        full_shows = server.fetchItems(show_keys) if show_keys else []
        for full_show in full_shows:
            show_ids = _extract_external_ids(full_show)
            show_tmdb: Optional[TMDBShow] = None
            if show_ids.tmdb:
                show_tmdb = await _fetch_tmdb_show(client, show_ids.tmdb, tmdb_api_key)
            episode_keys = [int(e.ratingKey) for e in full_show.episodes()]
            episodes = server.fetchItems(episode_keys) if episode_keys else []
            ep_imdb_ids = [
                _extract_external_ids(e).imdb
                for e in episodes
                if _extract_external_ids(e).imdb
            ]
            ep_imdb_map = (
                await _fetch_imdb_batch(client, ep_imdb_ids) if ep_imdb_ids else {}
            )

            async def _augment_episode(episode: PlexPartialObject) -> AggregatedItem:
                ids = _extract_external_ids(episode)
                imdb = ep_imdb_map.get(ids.imdb) if ids.imdb else None
                season = resolve_tmdb_season_number(show_tmdb, episode)
                ep_num = getattr(episode, "index", None)
                tmdb_episode = (
                    await _fetch_tmdb_episode(
                        client, show_tmdb.id, season, ep_num, tmdb_api_key
                    )
                    if show_tmdb and season is not None and ep_num is not None
                    else None
                )
                tmdb: Optional[TMDBItem] = tmdb_episode or show_tmdb
                return AggregatedItem(
                    plex=_build_plex_item(episode), imdb=imdb, tmdb=tmdb
                )

            episode_tasks = [_augment_episode(ep) for ep in episodes]
            if episode_tasks:
                async for result in _iter_gather_in_batches(episode_tasks, batch_size):
                    yield result


async def _load_from_plex(
    server: PlexServer, tmdb_api_key: str, *, batch_size: int = 50
) -> List[AggregatedItem]:
    """Retain list-based API for tests by consuming :func:`_iter_from_plex`."""

    return [item async for item in _iter_from_plex(server, tmdb_api_key, batch_size=batch_size)]


def _load_from_sample(sample_dir: Path) -> List[AggregatedItem]:
    """Load items from local sample JSON files."""

    results: List[AggregatedItem] = []
    movie_dir = sample_dir / "movie"
    episode_dir = sample_dir / "episode"

    # Movie sample
    with (movie_dir / "plex.json").open("r", encoding="utf-8") as f:
        movie_data = json.load(f)["MediaContainer"]["Metadata"][0]
    plex_movie = PlexItem(
        rating_key=str(movie_data.get("ratingKey", "")),
        guid=str(movie_data.get("guid", "")),
        type=movie_data.get("type", "movie"),
        title=movie_data.get("title", ""),
        summary=movie_data.get("summary"),
        year=movie_data.get("year"),
        added_at=movie_data.get("addedAt"),
        guids=[PlexGuid(id=g["id"]) for g in movie_data.get("Guid", [])],
        thumb=movie_data.get("thumb"),
        art=movie_data.get("art"),
        tagline=movie_data.get("tagline"),
        content_rating=movie_data.get("contentRating"),
        directors=[
            PlexPerson(id=d.get("id", 0), tag=d.get("tag", ""), thumb=d.get("thumb"))
            for d in movie_data.get("Director", [])
        ],
        writers=[
            PlexPerson(id=w.get("id", 0), tag=w.get("tag", ""), thumb=w.get("thumb"))
            for w in movie_data.get("Writer", [])
        ],
        actors=[
            PlexPerson(
                id=a.get("id", 0),
                tag=a.get("tag", ""),
                role=a.get("role"),
                thumb=a.get("thumb"),
            )
            for a in movie_data.get("Role", [])
        ],
        genres=[g.get("tag", "") for g in movie_data.get("Genre", []) if g.get("tag")],
        collections=[
            c.get("tag", "")
            for key in ("Collection", "Collections")
            for c in movie_data.get(key, []) or []
            if c.get("tag")
        ],
    )
    with (movie_dir / "imdb.json").open("r", encoding="utf-8") as f:
        imdb_movie = IMDbTitle.model_validate(json.load(f))
    with (movie_dir / "tmdb.json").open("r", encoding="utf-8") as f:
        tmdb_movie = TMDBMovie.model_validate(json.load(f))
    results.append(AggregatedItem(plex=plex_movie, imdb=imdb_movie, tmdb=tmdb_movie))

    # Episode sample
    with (episode_dir / "plex.tv.json").open("r", encoding="utf-8") as f:
        episode_data = json.load(f)["MediaContainer"]["Metadata"][0]
    plex_episode = PlexItem(
        rating_key=str(episode_data.get("ratingKey", "")),
        guid=str(episode_data.get("guid", "")),
        type=episode_data.get("type", "episode"),
        title=episode_data.get("title", ""),
        show_title=episode_data.get("grandparentTitle"),
        season_title=episode_data.get("parentTitle"),
        season_number=episode_data.get("parentIndex"),
        episode_number=episode_data.get("index"),
        summary=episode_data.get("summary"),
        year=episode_data.get("year"),
        added_at=episode_data.get("addedAt"),
        guids=[PlexGuid(id=g["id"]) for g in episode_data.get("Guid", [])],
        thumb=episode_data.get("thumb"),
        art=episode_data.get("art"),
        tagline=episode_data.get("tagline"),
        content_rating=episode_data.get("contentRating"),
        directors=[
            PlexPerson(id=d.get("id", 0), tag=d.get("tag", ""), thumb=d.get("thumb"))
            for d in episode_data.get("Director", [])
        ],
        writers=[
            PlexPerson(id=w.get("id", 0), tag=w.get("tag", ""), thumb=w.get("thumb"))
            for w in episode_data.get("Writer", [])
        ],
        actors=[
            PlexPerson(
                id=a.get("id", 0),
                tag=a.get("tag", ""),
                role=a.get("role"),
                thumb=a.get("thumb"),
            )
            for a in episode_data.get("Role", [])
        ],
        genres=[g.get("tag", "") for g in episode_data.get("Genre", []) if g.get("tag")],
        collections=[
            c.get("tag", "")
            for key in ("Collection", "Collections")
            for c in episode_data.get(key, []) or []
            if c.get("tag")
        ],
    )
    with (episode_dir / "imdb.tv.json").open("r", encoding="utf-8") as f:
        imdb_episode = IMDbTitle.model_validate(json.load(f))
    with (episode_dir / "tmdb.tv.json").open("r", encoding="utf-8") as f:
        tmdb_show = TMDBShow.model_validate(json.load(f))
    results.append(AggregatedItem(plex=plex_episode, imdb=imdb_episode, tmdb=tmdb_show))

    return results


async def _iter_from_sample(sample_dir: Path) -> AsyncIterator[AggregatedItem]:
    """Yield sample data items for streaming pipelines."""

    for item in _load_from_sample(sample_dir):
        yield item


async def run(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    qdrant_host: Optional[str] = None,
    qdrant_port: int = 6333,
    qdrant_grpc_port: int = 6334,
    qdrant_https: bool = False,
    qdrant_prefer_grpc: bool = False,
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
    sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
    imdb_cache_path: Path | None = None,
    imdb_max_retries: int = 3,
    imdb_backoff: float = 1.0,
    imdb_queue_path: Path | None = None,
    upsert_buffer_size: int = _qdrant_upsert_buffer_size,
) -> None:
    """Core execution logic for the CLI."""

    global _imdb_cache, _imdb_max_retries, _imdb_backoff, _imdb_retry_queue
    _imdb_cache = IMDbCache(imdb_cache_path) if imdb_cache_path else None
    _imdb_max_retries = imdb_max_retries
    _imdb_backoff = imdb_backoff
    if imdb_queue_path:
        _load_imdb_retry_queue(imdb_queue_path)
        async with httpx.AsyncClient(timeout=30) as client:
            await _process_imdb_retry_queue(client)
    else:
        _imdb_retry_queue = _IMDbRetryQueue()

    _require_positive(upsert_buffer_size, name="upsert_buffer_size")

    dense_size, dense_distance = _resolve_dense_model_params(dense_model_name)
    if qdrant_url is None and qdrant_host is None:
        qdrant_url = ":memory:"
    client = AsyncQdrantClient(
        location=qdrant_url,
        api_key=qdrant_api_key,
        host=qdrant_host,
        port=qdrant_port,
        grpc_port=qdrant_grpc_port,
        https=qdrant_https,
        prefer_grpc=qdrant_prefer_grpc,
    )
    collection_name = "media-items"
    await _ensure_collection(
        client,
        collection_name,
        dense_size=dense_size,
        dense_distance=dense_distance,
    )

    items: List[AggregatedItem] = []
    if sample_dir is not None:
        logger.info("Loading sample data from %s", sample_dir)
        item_iter = _iter_from_sample(sample_dir)
    else:
        if PlexServer is None:
            raise RuntimeError("plexapi is required for live loading")
        if not plex_url or not plex_token:
            raise RuntimeError("PLEX_URL and PLEX_TOKEN must be provided")
        if not tmdb_api_key:
            raise RuntimeError("TMDB_API_KEY must be provided")
        logger.info("Loading data from Plex server %s", plex_url)
        server = PlexServer(plex_url, plex_token)
        item_iter = _iter_from_plex(server, tmdb_api_key)

    points_buffer: List[models.PointStruct] = []
    upsert_tasks: list[asyncio.Task[None]] = []
    max_concurrent_upserts = _require_positive(
        _qdrant_max_concurrent_upserts, name="max_concurrent_upserts"
    )
    upsert_semaphore = asyncio.Semaphore(max_concurrent_upserts)

    async def _run_upsert(batch: List[models.PointStruct]) -> None:
        async with upsert_semaphore:
            await _upsert_in_batches(
                client,
                collection_name,
                batch,
            )

    def _schedule_upsert(batch: List[models.PointStruct]) -> None:
        logger.info(
            "Upserting %d points into Qdrant collection %s in batches of %d",
            len(batch),
            collection_name,
            _qdrant_batch_size,
        )
        upsert_tasks.append(asyncio.create_task(_run_upsert(batch)))

    async for item in item_iter:
        items.append(item)
        primary_title = item.plex.title
        if item.plex.type == "episode":
            title_bits: list[str] = []
            if item.plex.show_title:
                title_bits.append(item.plex.show_title)
            se_parts: list[str] = []
            if item.plex.season_number is not None:
                se_parts.append(f"S{item.plex.season_number:02d}")
            if item.plex.episode_number is not None:
                se_parts.append(f"E{item.plex.episode_number:02d}")
            if se_parts:
                title_bits.append("".join(se_parts))
            if item.plex.title:
                title_bits.append(item.plex.title)
            if title_bits:
                primary_title = " - ".join(title_bits)
        parts = [
            primary_title,
            item.plex.summary or "",
            item.tmdb.overview if item.tmdb and hasattr(item.tmdb, "overview") else "",
            item.imdb.plot if item.imdb else "",
        ]
        directors_text = ", ".join(p.tag for p in item.plex.directors if p.tag)
        writers_text = ", ".join(p.tag for p in item.plex.writers if p.tag)
        actors_text = ", ".join(p.tag for p in item.plex.actors if p.tag)
        if directors_text:
            parts.append(f"Directed by {directors_text}")
        if writers_text:
            parts.append(f"Written by {writers_text}")
        if actors_text:
            parts.append(f"Starring {actors_text}")
        if item.plex.tagline:
            parts.append(item.plex.tagline)
        if item.tmdb and hasattr(item.tmdb, "tagline"):
            tagline = getattr(item.tmdb, "tagline", None)
            if tagline:
                parts.append(tagline)
        if item.tmdb and hasattr(item.tmdb, "reviews"):
            parts.extend(r.get("content", "") for r in getattr(item.tmdb, "reviews", []))
        text = "\n".join(p for p in parts if p)
        payload = {
            "data": item.model_dump(mode="json"),
            "title": item.plex.title,
            "type": item.plex.type,
        }
        if item.plex.type == "episode":
            if item.plex.show_title:
                payload["show_title"] = item.plex.show_title
            if item.plex.season_title:
                payload["season_title"] = item.plex.season_title
            if item.plex.season_number is not None:
                payload["season_number"] = item.plex.season_number
            if item.plex.episode_number is not None:
                payload["episode_number"] = item.plex.episode_number
        if item.plex.actors:
            payload["actors"] = [p.tag for p in item.plex.actors if p.tag]
        if item.plex.directors:
            payload["directors"] = [p.tag for p in item.plex.directors if p.tag]
        if item.plex.writers:
            payload["writers"] = [p.tag for p in item.plex.writers if p.tag]
        if item.plex.genres:
            payload["genres"] = item.plex.genres
        if item.plex.collections:
            payload["collections"] = item.plex.collections
        summary = item.plex.summary
        if summary:
            payload["summary"] = summary
        overview = getattr(item.tmdb, "overview", None) if item.tmdb else None
        if overview:
            payload["overview"] = overview
        plot = item.imdb.plot if item.imdb else None
        if plot:
            payload["plot"] = plot
        taglines = [item.plex.tagline]
        if item.tmdb and hasattr(item.tmdb, "tagline"):
            taglines.append(getattr(item.tmdb, "tagline", None))
        taglines = [t for t in taglines if t]
        if taglines:
            payload["tagline"] = "\n".join(dict.fromkeys(taglines))
        if item.tmdb and hasattr(item.tmdb, "reviews"):
            review_texts = [r.get("content", "") for r in getattr(item.tmdb, "reviews", [])]
            review_texts = [r for r in review_texts if r]
            if review_texts:
                payload["reviews"] = review_texts
        if item.plex.year is not None:
            payload["year"] = item.plex.year
        if item.plex.added_at is not None:
            payload["added_at"] = int(item.plex.added_at.timestamp())
        point_id: int | str = (
            int(item.plex.rating_key)
            if item.plex.rating_key.isdigit()
            else item.plex.rating_key
        )
        points_buffer.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": models.Document(text=text, model=dense_model_name),
                    "sparse": models.Document(text=text, model=sparse_model_name),
                },
                payload=payload,
            )
        )

        if len(points_buffer) >= upsert_buffer_size:
            batch = list(points_buffer)
            points_buffer.clear()
            _schedule_upsert(batch)

    logger.info("Loaded %d items", len(items))

    if points_buffer:
        batch = list(points_buffer)
        points_buffer.clear()
        _schedule_upsert(batch)

    if upsert_tasks:
        await asyncio.gather(*upsert_tasks)
    else:
        logger.info("No points to upsert")

    if imdb_queue_path:
        _persist_imdb_retry_queue(imdb_queue_path)

    json.dump([item.model_dump(mode="json") for item in items], fp=sys.stdout, indent=2)
    sys.stdout.write("\n")


@click.command()
@click.option(
    "--plex-url",
    envvar="PLEX_URL",
    show_envvar=True,
    required=True,
    help="Plex base URL",
)
@click.option(
    "--plex-token",
    envvar="PLEX_TOKEN",
    show_envvar=True,
    required=True,
    help="Plex API token",
)
@click.option(
    "--tmdb-api-key",
    envvar="TMDB_API_KEY",
    show_envvar=True,
    required=True,
    help="TMDb API key",
)
@click.option(
    "--sample-dir",
    type=click.Path(path_type=Path),
    required=False,
    help="Directory containing sample data instead of live Plex access",
)
@click.option(
    "--qdrant-url",
    envvar="QDRANT_URL",
    show_envvar=True,
    required=False,
    help="Qdrant URL or path",
)
@click.option(
    "--qdrant-api-key",
    envvar="QDRANT_API_KEY",
    show_envvar=True,
    required=False,
    help="Qdrant API key",
)
@click.option(
    "--qdrant-host",
    envvar="QDRANT_HOST",
    show_envvar=True,
    required=False,
    help="Qdrant host",
)
@click.option(
    "--qdrant-port",
    envvar="QDRANT_PORT",
    show_envvar=True,
    type=int,
    default=6333,
    show_default=True,
    required=False,
    help="Qdrant HTTP port",
)
@click.option(
    "--qdrant-grpc-port",
    envvar="QDRANT_GRPC_PORT",
    show_envvar=True,
    type=int,
    default=6334,
    show_default=True,
    required=False,
    help="Qdrant gRPC port",
)
@click.option(
    "--qdrant-https/--no-qdrant-https",
    envvar="QDRANT_HTTPS",
    show_envvar=True,
    default=False,
    help="Use HTTPS when connecting to Qdrant",
)
@click.option(
    "--qdrant-prefer-grpc/--no-qdrant-prefer-grpc",
    envvar="QDRANT_PREFER_GRPC",
    show_envvar=True,
    default=False,
    help="Prefer gRPC when connecting to Qdrant",
)
@click.option(
    "--upsert-buffer-size",
    envvar="QDRANT_UPSERT_BUFFER_SIZE",
    show_envvar=True,
    type=int,
    default=_qdrant_upsert_buffer_size,
    show_default=True,
    help="Number of media items to buffer before scheduling an async upsert",
)
@click.option(
    "--dense-model",
    envvar="DENSE_MODEL",
    show_envvar=True,
    default="BAAI/bge-small-en-v1.5",
    show_default=True,
    help="Dense embedding model name",
)
@click.option(
    "--sparse-model",
    envvar="SPARSE_MODEL",
    show_envvar=True,
    default="Qdrant/bm42-all-minilm-l6-v2-attentions",
    show_default=True,
    help="Sparse embedding model name",
)
@click.option(
    "--continuous",
    is_flag=True,
    help="Continuously run the loader",
    show_default=True,
    default=False,
    required=False,
)
@click.option(
    "--delay",
    type=float,
    default=300.0,
    show_default=True,
    required=False,
    help="Delay between runs in seconds when using --continuous",
)
@click.option(
    "--imdb-cache",
    envvar="IMDB_CACHE",
    show_envvar=True,
    type=click.Path(path_type=Path),
    default=Path("imdb_cache.json"),
    show_default=True,
    help="Path to persistent IMDb response cache",
)
@click.option(
    "--imdb-max-retries",
    envvar="IMDB_MAX_RETRIES",
    show_envvar=True,
    type=int,
    default=3,
    show_default=True,
    help="Maximum retries for IMDb requests returning HTTP 429",
)
@click.option(
    "--imdb-backoff",
    envvar="IMDB_BACKOFF",
    show_envvar=True,
    type=float,
    default=1.0,
    show_default=True,
    help="Initial backoff delay in seconds for IMDb retries",
)
@click.option(
    "--imdb-queue",
    envvar="IMDB_QUEUE",
    show_envvar=True,
    type=click.Path(path_type=Path),
    default=Path("imdb_queue.json"),
    show_default=True,
    help="Path to persistent IMDb retry queue",
)
def main(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    qdrant_host: Optional[str],
    qdrant_port: int,
    qdrant_grpc_port: int,
    qdrant_https: bool,
    qdrant_prefer_grpc: bool,
    upsert_buffer_size: int,
    dense_model: str,
    sparse_model: str,
    continuous: bool,
    delay: float,
    imdb_cache: Path,
    imdb_max_retries: int,
    imdb_backoff: float,
    imdb_queue: Path,
) -> None:
    """Entry-point for the ``load-data`` script."""

    asyncio.run(
        load_media(
            plex_url,
            plex_token,
            tmdb_api_key,
            sample_dir,
            qdrant_url,
            qdrant_api_key,
            qdrant_host,
            qdrant_port,
            qdrant_grpc_port,
            qdrant_https,
            qdrant_prefer_grpc,
            dense_model,
            sparse_model,
            continuous,
            delay,
            imdb_cache,
            imdb_max_retries,
            imdb_backoff,
            imdb_queue,
            upsert_buffer_size,
        )
    )


async def load_media(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    qdrant_host: Optional[str],
    qdrant_port: int,
    qdrant_grpc_port: int,
    qdrant_https: bool,
    qdrant_prefer_grpc: bool,
    dense_model_name: str,
    sparse_model_name: str,
    continuous: bool,
    delay: float,
    imdb_cache: Path,
    imdb_max_retries: int,
    imdb_backoff: float,
    imdb_queue: Path,
    upsert_buffer_size: int,
) -> None:
    """Orchestrate one or more runs of :func:`run`."""

    while True:
        await run(
            plex_url,
            plex_token,
            tmdb_api_key,
            sample_dir,
            qdrant_url,
            qdrant_api_key,
            qdrant_host,
            qdrant_port,
            qdrant_grpc_port,
            qdrant_https,
            qdrant_prefer_grpc,
            dense_model_name,
            sparse_model_name,
            imdb_cache,
            imdb_max_retries,
            imdb_backoff,
            imdb_queue,
            upsert_buffer_size,
        )
        if not continuous:
            break

        await asyncio.sleep(delay)


if __name__ == "__main__":
    main()
