"""Utilities for loading Plex metadata with IMDb and TMDb details."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

import click
import httpx
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client import models

from .types import (
    AggregatedItem,
    ExternalIDs,
    IMDbTitle,
    PlexItem,
    TMDBEpisode,
    TMDBItem,
    TMDBMovie,
    TMDBShow,
    PlexGuid,
    PlexPerson,
)

try:  # Only import plexapi when available; the sample data mode does not require it.
    from plexapi.server import PlexServer
    from plexapi.base import PlexPartialObject
except Exception:
    PlexServer = None  # type: ignore[assignment]
    PlexPartialObject = object  # type: ignore[assignment]


async def _fetch_imdb(client: httpx.AsyncClient, imdb_id: str) -> Optional[IMDbTitle]:
    """Fetch metadata for an IMDb ID."""

    url = f"https://api.imdbapi.dev/titles/{imdb_id}"
    resp = await client.get(url)
    if resp.is_success:
        return IMDbTitle.model_validate(resp.json())
    return None


async def _fetch_tmdb_movie(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBMovie]:
    url = (
        f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&append_to_response=reviews"
    )
    resp = await client.get(url)
    if resp.is_success:
        return TMDBMovie.model_validate(resp.json())
    return None


async def _fetch_tmdb_show(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBShow]:
    url = (
        f"https://api.themoviedb.org/3/tv/{tmdb_id}?api_key={api_key}&append_to_response=reviews"
    )
    resp = await client.get(url)
    if resp.is_success:
        return TMDBShow.model_validate(resp.json())
    return None


async def _fetch_tmdb_episode(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBEpisode]:
    """Attempt to fetch TMDb data for a TV episode by its ID."""

    url = f"https://api.themoviedb.org/3/tv/episode/{tmdb_id}?api_key={api_key}"
    resp = await client.get(url)
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
        guids=guids,
        thumb=getattr(item, "thumb", None),
        art=getattr(item, "art", None),
        tagline=getattr(item, "tagline", None),
        content_rating=getattr(item, "contentRating", None),
        directors=directors,
        writers=writers,
        actors=actors,
    )


async def _load_from_plex(server: PlexServer, tmdb_api_key: str) -> List[AggregatedItem]:
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
        tmdb_task = (
            _fetch_tmdb_episode(client, ids.tmdb, tmdb_api_key)
            if ids.tmdb
            else asyncio.sleep(0, result=None)
        )
        imdb, tmdb_episode = await asyncio.gather(imdb_task, tmdb_task)
        tmdb: Optional[TMDBItem] = tmdb_episode or show_tmdb
        return AggregatedItem(plex=_build_plex_item(episode), imdb=imdb, tmdb=tmdb)

    results: List[AggregatedItem] = []
    async with httpx.AsyncClient(timeout=30) as client:
        movie_section = server.library.section("Movies")
        movie_tasks = [
            _augment_movie(client, movie) for movie in movie_section.all()
        ]
        if movie_tasks:
            results.extend(await asyncio.gather(*movie_tasks))

        show_section = server.library.section("TV Shows")
        for show in show_section.all():
            show_ids = _extract_external_ids(show)
            show_tmdb: Optional[TMDBShow] = None
            if show_ids.tmdb:
                show_tmdb = await _fetch_tmdb_show(client, show_ids.tmdb, tmdb_api_key)
            episode_tasks = [
                _augment_episode(client, episode, show_tmdb) for episode in show.episodes()
            ]
            if episode_tasks:
                results.extend(await asyncio.gather(*episode_tasks))
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
) -> None:
    """Core execution logic for the CLI."""

    items: List[AggregatedItem]
    if sample_dir is not None:
        items = _load_from_sample(sample_dir)
    else:
        if PlexServer is None:
            raise RuntimeError("plexapi is required for live loading")
        if not plex_url or not plex_token:
            raise RuntimeError("PLEX_URL and PLEX_TOKEN must be provided")
        if not tmdb_api_key:
            raise RuntimeError("TMDB_API_KEY must be provided")
        server = PlexServer(plex_url, plex_token)
        items = await _load_from_plex(server, tmdb_api_key)

    # Embed and store in Qdrant
    texts: List[str] = []
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
        texts.append("\n".join(p for p in parts if p))

    dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
    sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

    dense_vectors = list(dense_model.embed(texts))
    sparse_vectors = list(sparse_model.passage_embed(texts))

    client = AsyncQdrantClient(qdrant_url or ":memory:", api_key=qdrant_api_key)
    collection_name = "media-items"
    vectors_config = {
        "dense": models.VectorParams(
            size=dense_model.embedding_size, distance=models.Distance.COSINE
        )
    }
    sparse_vectors_config = {"sparse": models.SparseVectorParams()}

    created_collection = False
    if await client.collection_exists(collection_name):
        info = await client.get_collection(collection_name)
        existing_size = info.config.params.vectors["dense"].size  # type: ignore[index]
        if existing_size != dense_model.embedding_size:
            await client.delete_collection(collection_name)
            await client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
            created_collection = True
    else:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        created_collection = True

    if created_collection:
        await client.create_payload_index(
            collection_name=collection_name,
            field_name="search_text",
            field_schema=models.TextIndexParams(
                type=models.PayloadSchemaType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                lowercase=True,
            ),
        )
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

    points = []
    for idx, (item, dense, sparse) in enumerate(zip(items, dense_vectors, sparse_vectors)):
        sv = models.SparseVector(
            indices=sparse.indices.tolist(), values=sparse.values.tolist()
        )
        payload = {
            "data": item.model_dump(),
            "search_text": texts[idx],
            "title": item.plex.title,
            "type": item.plex.type,
        }
        if item.plex.year is not None:
            payload["year"] = item.plex.year
        points.append(
            models.Record(
                id=int(item.plex.rating_key)
                if item.plex.rating_key.isdigit()
                else item.plex.rating_key,
                payload=payload,
                vector={"dense": dense, "sparse": sv},
            )
        )

    if points:
        await client.upsert(collection_name="media-items", points=points)

    json.dump([item.model_dump() for item in items], fp=sys.stdout, indent=2)
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
def main(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    continuous: bool,
    delay: float,
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
            continuous,
            delay,
        )
    )


async def load_media(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    continuous: bool,
    delay: float,
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
        )
        if not continuous:
            break

        await asyncio.sleep(delay)


if __name__ == "__main__":
    main()
