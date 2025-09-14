"""Utilities for loading Plex metadata with IMDb and TMDb details."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Awaitable, List, Optional, Sequence, TypeVar

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


logger = logging.getLogger(__name__)

T = TypeVar("T")

_imdb_cache: IMDbCache | None = None
_imdb_max_retries: int = 3
_imdb_backoff: float = 1.0
_imdb_retry_queue: asyncio.Queue[str] | None = None


async def _gather_in_batches(
    tasks: Sequence[Awaitable[T]], batch_size: int
) -> List[T]:
    """Gather awaitable tasks in fixed-size batches."""

    results: List[T] = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        results.extend(await asyncio.gather(*batch))
    return results


async def _fetch_imdb(client: httpx.AsyncClient, imdb_id: str) -> Optional[IMDbTitle]:
    """Fetch metadata for an IMDb ID with caching and retry logic."""

    if _imdb_cache:
        cached = _imdb_cache.get(imdb_id)
        if cached:
            return IMDbTitle.model_validate(cached)

    url = f"https://api.imdbapi.dev/titles/{imdb_id}"
    delay = _imdb_backoff
    for attempt in range(_imdb_max_retries + 1):
        resp = await client.get(url)
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


def _load_imdb_retry_queue(path: Path) -> None:
    """Populate the retry queue from a JSON file if it exists."""

    global _imdb_retry_queue
    _imdb_retry_queue = asyncio.Queue()
    if path.exists():
        try:
            ids = json.loads(path.read_text())
            for imdb_id in ids:
                _imdb_retry_queue.put_nowait(str(imdb_id))
        except Exception:
            logger.exception("Failed to load IMDb retry queue from %s", path)


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
    ids = list(_imdb_retry_queue._queue)  # type: ignore[attr-defined]
    path.write_text(json.dumps(ids))


async def _fetch_tmdb_movie(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBMovie]:
    url = (
        f"https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=reviews"
    )
    resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    if resp.is_success:
        return TMDBMovie.model_validate(resp.json())
    return None


async def _fetch_tmdb_show(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBShow]:
    url = (
        f"https://api.themoviedb.org/3/tv/{tmdb_id}?append_to_response=reviews"
    )
    resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
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
    resp = await client.get(url, headers={"Authorization": f"Bearer {api_key}"})
    if resp.is_success:
        return TMDBEpisode.model_validate(resp.json())
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
    return PlexItem(
        rating_key=str(getattr(item, "ratingKey", "")),
        guid=str(getattr(item, "guid", "")),
        type=str(getattr(item, "type", "")),
        title=str(getattr(item, "title", "")),
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
    )


async def _load_from_plex(
    server: PlexServer, tmdb_api_key: str, *, batch_size: int = 50
) -> List[AggregatedItem]:
    """Load items from a live Plex server."""

    async def _augment_movie(client: httpx.AsyncClient, movie: PlexPartialObject) -> AggregatedItem:
        ids = _extract_external_ids(movie)
        imdb_task = (
            _fetch_imdb(client, ids.imdb) if ids.imdb else asyncio.sleep(0, result=None)
        )
        tmdb_task = (
            _fetch_tmdb_movie(client, ids.tmdb, tmdb_api_key)
            if ids.tmdb
            else asyncio.sleep(0, result=None)
        )
        imdb, tmdb = await asyncio.gather(imdb_task, tmdb_task)
        return AggregatedItem(plex=_build_plex_item(movie), imdb=imdb, tmdb=tmdb)

    async def _augment_episode(
        client: httpx.AsyncClient,
        episode: PlexPartialObject,
        show_tmdb: Optional[TMDBShow],
    ) -> AggregatedItem:
        ids = _extract_external_ids(episode)
        imdb_task = (
            _fetch_imdb(client, ids.imdb) if ids.imdb else asyncio.sleep(0, result=None)
        )
        season = getattr(episode, "parentIndex", None)
        if season is None:
            title = getattr(episode, "parentTitle", "")
            if isinstance(title, str) and title.isdigit():
                season = int(title)
        ep_num = getattr(episode, "index", None)
        tmdb_task = (
            _fetch_tmdb_episode(client, show_tmdb.id, season, ep_num, tmdb_api_key)
            if show_tmdb and season is not None and ep_num is not None
            else asyncio.sleep(0, result=None)
        )
        imdb, tmdb_episode = await asyncio.gather(imdb_task, tmdb_task)
        tmdb: Optional[TMDBItem] = tmdb_episode or show_tmdb
        return AggregatedItem(plex=_build_plex_item(episode), imdb=imdb, tmdb=tmdb)

    results: List[AggregatedItem] = []
    async with httpx.AsyncClient(timeout=30) as client:
        movie_section = server.library.section("Movies")
        movie_tasks = [
            _augment_movie(client, movie.fetchItem(movie.ratingKey))
            for movie in movie_section.all()
        ]
        if movie_tasks:
            results.extend(await _gather_in_batches(movie_tasks, batch_size))

        show_section = server.library.section("TV Shows")
        for show in show_section.all():
            full_show = show.fetchItem(show.ratingKey)
            show_ids = _extract_external_ids(full_show)
            show_tmdb: Optional[TMDBShow] = None
            if show_ids.tmdb:
                show_tmdb = await _fetch_tmdb_show(client, show_ids.tmdb, tmdb_api_key)
            episode_tasks = [
                _augment_episode(
                    client, episode.fetchItem(episode.ratingKey), show_tmdb
                )
                for episode in full_show.episodes()
            ]
            if episode_tasks:
                results.extend(await _gather_in_batches(episode_tasks, batch_size))
    return results


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
    )
    with (episode_dir / "imdb.tv.json").open("r", encoding="utf-8") as f:
        imdb_episode = IMDbTitle.model_validate(json.load(f))
    with (episode_dir / "tmdb.tv.json").open("r", encoding="utf-8") as f:
        tmdb_show = TMDBShow.model_validate(json.load(f))
    results.append(AggregatedItem(plex=plex_episode, imdb=imdb_episode, tmdb=tmdb_show))

    return results


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
        _imdb_retry_queue = asyncio.Queue()

    items: List[AggregatedItem]
    if sample_dir is not None:
        logger.info("Loading sample data from %s", sample_dir)
        items = _load_from_sample(sample_dir)
    else:
        if PlexServer is None:
            raise RuntimeError("plexapi is required for live loading")
        if not plex_url or not plex_token:
            raise RuntimeError("PLEX_URL and PLEX_TOKEN must be provided")
        if not tmdb_api_key:
            raise RuntimeError("TMDB_API_KEY must be provided")
        logger.info("Loading data from Plex server %s", plex_url)
        server = PlexServer(plex_url, plex_token)
        items = await _load_from_plex(server, tmdb_api_key)
    logger.info("Loaded %d items", len(items))

    # Assemble points with server-side embeddings
    points: List[models.PointStruct] = []
    for item in items:
        parts = [
            item.plex.title,
            item.plex.summary or "",
            item.tmdb.overview if item.tmdb and hasattr(item.tmdb, "overview") else "",
            item.imdb.plot if item.imdb else "",
            " ".join(p.tag for p in item.plex.directors),
            " ".join(p.tag for p in item.plex.writers),
            " ".join(p.tag for p in item.plex.actors),
        ]
        if item.tmdb and hasattr(item.tmdb, "reviews"):
            parts.extend(r.get("content", "") for r in getattr(item.tmdb, "reviews", []))
        text = "\n".join(p for p in parts if p)
        payload = {
            "data": item.model_dump(mode="json"),
            "title": item.plex.title,
            "type": item.plex.type,
        }
        if item.plex.actors:
            payload["actors"] = [p.tag for p in item.plex.actors]
        if item.plex.year is not None:
            payload["year"] = item.plex.year
        if item.plex.added_at is not None:
            payload["added_at"] = int(item.plex.added_at.timestamp())
        point_id: int | str = (
            int(item.plex.rating_key)
            if item.plex.rating_key.isdigit()
            else item.plex.rating_key
        )
        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": models.Document(text=text, model=dense_model_name),
                    "sparse": models.Document(text=text, model=sparse_model_name),
                },
                payload=payload,
            )
        )

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
    dense_size, dense_distance = client._get_model_params(dense_model_name)
    collection_name = "media-items"
    created_collection = False
    if not await client.collection_exists(collection_name):
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": models.VectorParams(size=dense_size, distance=dense_distance)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        created_collection = True

    if created_collection:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="title",
            field_schema=models.TextIndexParams(
                type=models.PayloadSchemaType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                lowercase=True,
            ),
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

    if points:
        logger.info(
            "Upserting %d points into Qdrant collection %s",
            len(points),
            collection_name,
        )
        await client.upsert(collection_name=collection_name, points=points)
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
        )
        if not continuous:
            break

        await asyncio.sleep(delay)


if __name__ == "__main__":
    main()
