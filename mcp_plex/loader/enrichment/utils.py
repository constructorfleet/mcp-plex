from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Awaitable, Callable, Optional, Sequence

import httpx

from mcp_plex.common.types import (
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

from ..imdb_cache import IMDbCache
from ..utils import gather_in_batches
from .types import IMDbRetryQueue

try:
    from plexapi.base import PlexPartialObject
except Exception:  # pragma: no cover - fall back when plexapi unavailable
    PlexPartialObject = object  # type: ignore[assignment]


async def fetch_imdb(
    client: httpx.AsyncClient,
    imdb_id: str,
    *,
    cache: IMDbCache | None,
    max_retries: int,
    backoff: float,
    retry_queue: IMDbRetryQueue | None,
    logger: logging.Logger,
) -> Optional[IMDbTitle]:
    """Fetch metadata for an IMDb ID with caching and retry logic."""

    if cache:
        cached = cache.get(imdb_id)
        if cached:
            return IMDbTitle.model_validate(cached)

    url = f"https://api.imdbapi.dev/titles/{imdb_id}"
    delay = backoff
    for attempt in range(max_retries + 1):
        try:
            resp = await client.get(url)
        except httpx.HTTPError:
            logger.exception("HTTP error fetching IMDb ID %s", imdb_id)
            return None
        if resp.status_code == 429:
            if attempt == max_retries:
                if retry_queue is not None:
                    await retry_queue.put(imdb_id)
                return None
            await asyncio.sleep(delay)
            delay *= 2
            continue
        if resp.is_success:
            data = resp.json()
            if cache:
                cache.set(imdb_id, data)
            return IMDbTitle.model_validate(data)
        return None
    return None


async def fetch_imdb_batch(
    client: httpx.AsyncClient,
    imdb_ids: Sequence[str],
    *,
    cache: IMDbCache | None,
    batch_limit: int,
    max_retries: int,
    backoff: float,
    retry_queue: IMDbRetryQueue | None,
    logger: logging.Logger,
) -> dict[str, Optional[IMDbTitle]]:
    """Fetch metadata for multiple IMDb IDs, batching requests."""

    results: dict[str, Optional[IMDbTitle]] = {}
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
    for i in range(0, len(ids_to_fetch), batch_limit):
        chunk = ids_to_fetch[i : i + batch_limit]
        params = [("titleIds", imdb_id) for imdb_id in chunk]
        delay = backoff
        for attempt in range(max_retries + 1):
            try:
                resp = await client.get(url, params=params)
            except httpx.HTTPError:
                logger.exception("HTTP error fetching IMDb IDs %s", ",".join(chunk))
                for imdb_id in chunk:
                    results[imdb_id] = None
                break
            if resp.status_code == 429:
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
            if resp.is_success:
                data = resp.json()
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


def load_imdb_retry_queue(path: Path, logger: logging.Logger) -> IMDbRetryQueue:
    """Populate the retry queue from a JSON file if it exists."""

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
    return IMDbRetryQueue(ids)


async def process_imdb_retry_queue(
    client: httpx.AsyncClient,
    queue: IMDbRetryQueue,
    *,
    cache: IMDbCache | None,
    max_retries: int,
    backoff: float,
    logger: logging.Logger,
    fetch_fn: Callable[..., Awaitable[Optional[IMDbTitle]]] | None = None,
) -> None:
    """Attempt to fetch queued IMDb IDs, re-queueing failures."""

    if queue.empty():
        return
    size = queue.qsize()
    for _ in range(size):
        imdb_id = await queue.get()
        fetch = fetch_fn or fetch_imdb
        title = await fetch(
            client,
            imdb_id,
            cache=cache,
            max_retries=max_retries,
            backoff=backoff,
            retry_queue=queue,
            logger=logger,
        )
        if title is None:
            await queue.put(imdb_id)


def persist_imdb_retry_queue(path: Path, queue: IMDbRetryQueue) -> None:
    """Persist the retry queue to disk."""

    path.write_text(json.dumps(queue.snapshot()))


async def fetch_tmdb_movie(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str, logger: logging.Logger
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


async def fetch_tmdb_show(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str, logger: logging.Logger
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


async def fetch_tmdb_episode(
    client: httpx.AsyncClient,
    show_id: int,
    season_number: int,
    episode_number: int,
    api_key: str,
    logger: logging.Logger,
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


def extract_external_ids(item: PlexPartialObject) -> ExternalIDs:
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


def build_plex_item(item: PlexPartialObject) -> PlexItem:
    """Convert a Plex object into the internal :class:`PlexItem`."""

    guids = [PlexGuid(id=g.id) for g in getattr(item, "guids", [])]
    directors = [
        PlexPerson(
            id=getattr(d, "id", 0),
            tag=str(getattr(d, "tag", "")),
            thumb=getattr(d, "thumb", None),
        )
        for d in getattr(item, "directors", []) or []
    ]
    writers = [
        PlexPerson(
            id=getattr(w, "id", 0),
            tag=str(getattr(w, "tag", "")),
            thumb=getattr(w, "thumb", None),
        )
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


def resolve_tmdb_season_number(
    show_tmdb: Optional[TMDBShow], episode: PlexPartialObject
) -> Optional[int]:
    """Map a Plex episode to the appropriate TMDb season number."""

    parent_index = getattr(episode, "parentIndex", None)
    parent_title = getattr(episode, "parentTitle", None)
    parent_year = getattr(episode, "parentYear", None)
    if parent_year is None:
        parent_year = getattr(episode, "year", None)

    seasons = getattr(show_tmdb, "seasons", []) if show_tmdb else []

    if parent_index is not None:
        for season in seasons:
            if season.season_number == parent_index:
                return season.season_number

    title_norm: Optional[str] = None
    if isinstance(parent_title, str):
        title_norm = parent_title.lower().lstrip("season ").strip()
        for season in seasons:
            name_norm = (season.name or "").lower().lstrip("season ").strip()
            if name_norm == title_norm:
                return season.season_number

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


async def enrich_movies(
    client: httpx.AsyncClient,
    movies: Sequence[PlexPartialObject],
    *,
    tmdb_api_key: str | None,
    imdb_cache: IMDbCache | None,
    imdb_batch_limit: int,
    imdb_max_retries: int,
    imdb_backoff: float,
    imdb_retry_queue: IMDbRetryQueue | None,
    logger: logging.Logger,
) -> list[AggregatedItem]:
    """Enrich Plex movie metadata with IMDb and TMDb details."""

    movie_ids = [extract_external_ids(movie) for movie in movies]
    imdb_ids = [ids.imdb for ids in movie_ids if ids.imdb]
    imdb_map = (
        await fetch_imdb_batch(
            client,
            imdb_ids,
            cache=imdb_cache,
            batch_limit=imdb_batch_limit,
            max_retries=imdb_max_retries,
            backoff=imdb_backoff,
            retry_queue=imdb_retry_queue,
            logger=logger,
        )
        if imdb_ids
        else {}
    )

    tmdb_results: list[TMDBMovie | None] = []
    if tmdb_api_key:
        tmdb_tasks = [
            fetch_tmdb_movie(client, ids.tmdb, tmdb_api_key, logger)
            for ids in movie_ids
            if ids.tmdb
        ]
        if tmdb_tasks:
            tmdb_results = await gather_in_batches(tmdb_tasks, len(tmdb_tasks))
    tmdb_iter = iter(tmdb_results)

    aggregated: list[AggregatedItem] = []
    for movie, ids in zip(movies, movie_ids):
        tmdb = next(tmdb_iter, None) if ids.tmdb else None
        imdb = imdb_map.get(ids.imdb) if ids.imdb else None
        aggregated.append(
            AggregatedItem(
                plex=build_plex_item(movie),
                imdb=imdb,
                tmdb=tmdb,
            )
        )
    return aggregated


async def enrich_episodes(
    client: httpx.AsyncClient,
    show: PlexPartialObject,
    episodes: Sequence[PlexPartialObject],
    *,
    tmdb_api_key: str | None,
    imdb_cache: IMDbCache | None,
    imdb_batch_limit: int,
    imdb_max_retries: int,
    imdb_backoff: float,
    imdb_retry_queue: IMDbRetryQueue | None,
    show_tmdb_cache: dict[str, TMDBShow | None],
    logger: logging.Logger,
) -> list[AggregatedItem]:
    """Enrich Plex episode metadata with IMDb and TMDb details."""

    show_ids = extract_external_ids(show)
    show_tmdb: TMDBShow | None = None
    if show_ids.tmdb:
        if show_ids.tmdb in show_tmdb_cache:
            show_tmdb = show_tmdb_cache[show_ids.tmdb]
        elif tmdb_api_key:
            show_tmdb = await fetch_tmdb_show(
                client, show_ids.tmdb, tmdb_api_key, logger
            )
            show_tmdb_cache[show_ids.tmdb] = show_tmdb

    episode_ids = [extract_external_ids(ep) for ep in episodes]
    imdb_ids = [ids.imdb for ids in episode_ids if ids.imdb]
    imdb_map = (
        await fetch_imdb_batch(
            client,
            imdb_ids,
            cache=imdb_cache,
            batch_limit=imdb_batch_limit,
            max_retries=imdb_max_retries,
            backoff=imdb_backoff,
            retry_queue=imdb_retry_queue,
            logger=logger,
        )
        if imdb_ids
        else {}
    )

    tmdb_results: list[TMDBEpisode | None] = [None] * len(episodes)
    if show_tmdb and tmdb_api_key:
        episode_tasks: list[asyncio.Future[TMDBEpisode | None]] = []
        indices: list[int] = []
        for idx, ep in enumerate(episodes):
            season = resolve_tmdb_season_number(show_tmdb, ep)
            ep_num = getattr(ep, "index", None)
            if isinstance(ep_num, str) and ep_num.isdigit():
                ep_num = int(ep_num)
            if season is None or ep_num is None:
                continue
            indices.append(idx)
            episode_tasks.append(
                fetch_tmdb_episode(
                    client,
                    show_tmdb.id,
                    season,
                    ep_num,
                    tmdb_api_key,
                    logger,
                )
            )
        if episode_tasks:
            fetched = await gather_in_batches(episode_tasks, len(episode_tasks))
            for idx, value in zip(indices, fetched):
                tmdb_results[idx] = value

    aggregated: list[AggregatedItem] = []
    for ep, ids, tmdb_episode in zip(episodes, episode_ids, tmdb_results):
        imdb = imdb_map.get(ids.imdb) if ids.imdb else None
        tmdb_item: TMDBItem | None = tmdb_episode or show_tmdb
        aggregated.append(
            AggregatedItem(
                plex=build_plex_item(ep),
                imdb=imdb,
                tmdb=tmdb_item,
            )
        )
    return aggregated


__all__ = [
    "IMDbRetryQueue",
    "build_plex_item",
    "enrich_episodes",
    "enrich_movies",
    "extract_external_ids",
    "fetch_imdb",
    "fetch_imdb_batch",
    "fetch_tmdb_episode",
    "fetch_tmdb_movie",
    "fetch_tmdb_show",
    "load_imdb_retry_queue",
    "persist_imdb_retry_queue",
    "process_imdb_retry_queue",
    "resolve_tmdb_season_number",
]
