from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

from mcp_plex.common.types import (
    AggregatedItem,
    IMDbTitle,
    PlexGuid,
    PlexItem,
    PlexPerson,
    TMDBMovie,
    TMDBShow,
)

from ..utils import require_positive

T = TypeVar("T")


def chunk_sequence(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    """Yield ``items`` in chunks of at most ``size`` elements."""

    size = require_positive(int(size), name="size")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def load_from_sample(sample_dir: Path) -> list[AggregatedItem]:
    """Load items from local sample JSON files."""

    results: list[AggregatedItem] = []
    movie_dir = sample_dir / "movie"
    episode_dir = sample_dir / "episode"

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


__all__ = ["chunk_sequence", "load_from_sample"]
