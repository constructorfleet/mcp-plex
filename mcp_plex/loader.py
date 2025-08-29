"""Utilities for loading Plex metadata with IMDb and TMDb details."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import click
import httpx

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
)

try:  # Only import plexapi when available; the sample data mode does not require it.
    from plexapi.server import PlexServer
    from plexapi.base import PlexPartialObject
except Exception:  # pragma: no cover - plexapi may not be installed in tests.
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
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}"
    resp = await client.get(url)
    if resp.is_success:
        return TMDBMovie.model_validate(resp.json())
    return None


async def _fetch_tmdb_show(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBShow]:
    url = f"https://api.themoviedb.org/3/tv/{tmdb_id}?api_key={api_key}"
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
    return PlexItem(
        rating_key=str(getattr(item, "ratingKey", "")),
        guid=str(getattr(item, "guid", "")),
        type=str(getattr(item, "type", "")),
        title=str(getattr(item, "title", "")),
        summary=getattr(item, "summary", None),
        year=getattr(item, "year", None),
        guids=guids,
    )


async def _load_from_plex(server: PlexServer, tmdb_api_key: str) -> List[AggregatedItem]:
    """Load items from a live Plex server."""

    results: List[AggregatedItem] = []
    async with httpx.AsyncClient(timeout=30) as client:
        # Movies
        movie_section = server.library.section("Movies")
        for movie in movie_section.all():
            ids = _extract_external_ids(movie)
            imdb = await _fetch_imdb(client, ids.imdb) if ids.imdb else None
            tmdb = (
                await _fetch_tmdb_movie(client, ids.tmdb, tmdb_api_key)
                if ids.tmdb
                else None
            )
            results.append(
                AggregatedItem(plex=_build_plex_item(movie), imdb=imdb, tmdb=tmdb)
            )

        # TV Shows -> episodes
        show_section = server.library.section("TV Shows")
        for show in show_section.all():
            show_ids = _extract_external_ids(show)
            show_tmdb: Optional[TMDBShow] = None
            if show_ids.tmdb:
                show_tmdb = await _fetch_tmdb_show(client, show_ids.tmdb, tmdb_api_key)
            for episode in show.episodes():
                ids = _extract_external_ids(episode)
                imdb = await _fetch_imdb(client, ids.imdb) if ids.imdb else None
                tmdb: Optional[TMDBItem] = None
                if ids.tmdb:
                    tmdb = await _fetch_tmdb_episode(client, ids.tmdb, tmdb_api_key)
                if tmdb is None and show_tmdb is not None:
                    tmdb = show_tmdb
                results.append(
                    AggregatedItem(plex=_build_plex_item(episode), imdb=imdb, tmdb=tmdb)
                )
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

    json.dump([item.model_dump() for item in items], fp=os.sys.stdout, indent=2)
    os.sys.stdout.write("\n")


@click.command()
@click.option("--plex-url", envvar="PLEX_URL", help="Plex base URL")
@click.option("--plex-token", envvar="PLEX_TOKEN", help="Plex API token")
@click.option("--tmdb-api-key", envvar="TMDB_API_KEY", help="TMDb API key")
@click.option("--sample-dir", type=click.Path(path_type=Path))
def main(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
) -> None:
    """Entry-point for the ``load-data`` script."""

    asyncio.run(run(plex_url, plex_token, tmdb_api_key, sample_dir))


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
