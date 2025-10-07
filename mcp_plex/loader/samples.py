"""Helpers for working with built-in sample data files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..common.types import (
    AggregatedItem,
    IMDbTitle,
    PlexGuid,
    PlexItem,
    PlexPerson,
    TMDBMovie,
    TMDBShow,
)
from ..common.validation import coerce_plex_tag_id


def _read_json(path: Path) -> Any:
    """Return parsed JSON content from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_people(
    entries: Iterable[dict[str, Any]] | None,
    *,
    include_role: bool,
) -> list[PlexPerson]:
    """Construct :class:`PlexPerson` objects from Plex JSON entries."""

    people: list[PlexPerson] = []
    for entry in entries or []:
        person_kwargs: dict[str, Any] = {
            "id": coerce_plex_tag_id(entry.get("id", 0)),
            "tag": entry.get("tag", ""),
            "thumb": entry.get("thumb"),
        }
        if include_role:
            person_kwargs["role"] = entry.get("role")
        people.append(PlexPerson(**person_kwargs))
    return people


def _load_collections(data: dict[str, Any]) -> list[str]:
    """Extract collection tags from Plex metadata."""

    collections: list[str] = []
    for key in ("Collection", "Collections"):
        entries = data.get(key) or []
        for entry in entries:
            tag = entry.get("tag")
            if tag:
                collections.append(tag)
    return collections


def _load_plex_movie(data: dict[str, Any]) -> PlexItem:
    """Build a :class:`PlexItem` for the sample movie."""

    return PlexItem(
        rating_key=str(data.get("ratingKey", "")),
        guid=str(data.get("guid", "")),
        type=data.get("type", "movie"),
        title=data.get("title", ""),
        summary=data.get("summary"),
        year=data.get("year"),
        added_at=data.get("addedAt"),
        guids=[
            PlexGuid(id=str(guid.get("id", ""))) for guid in data.get("Guid", []) or []
        ],
        thumb=data.get("thumb"),
        art=data.get("art"),
        tagline=data.get("tagline"),
        content_rating=data.get("contentRating"),
        directors=_load_people(data.get("Director"), include_role=False),
        writers=_load_people(data.get("Writer"), include_role=False),
        actors=_load_people(data.get("Role"), include_role=True),
        genres=[
            genre.get("tag", "")
            for genre in data.get("Genre", []) or []
            if genre.get("tag")
        ],
        collections=_load_collections(data),
    )


def _load_plex_episode(data: dict[str, Any]) -> PlexItem:
    """Build a :class:`PlexItem` for the sample episode."""

    return PlexItem(
        rating_key=str(data.get("ratingKey", "")),
        guid=str(data.get("guid", "")),
        type=data.get("type", "episode"),
        title=data.get("title", ""),
        show_title=data.get("grandparentTitle"),
        season_title=data.get("parentTitle"),
        season_number=data.get("parentIndex"),
        episode_number=data.get("index"),
        summary=data.get("summary"),
        year=data.get("year"),
        added_at=data.get("addedAt"),
        guids=[
            PlexGuid(id=str(guid.get("id", ""))) for guid in data.get("Guid", []) or []
        ],
        thumb=data.get("thumb"),
        art=data.get("art"),
        tagline=data.get("tagline"),
        content_rating=data.get("contentRating"),
        directors=_load_people(data.get("Director"), include_role=False),
        writers=_load_people(data.get("Writer"), include_role=False),
        actors=_load_people(data.get("Role"), include_role=True),
        genres=[
            genre.get("tag", "")
            for genre in data.get("Genre", []) or []
            if genre.get("tag")
        ],
        collections=_load_collections(data),
    )


def _load_from_sample(sample_dir: Path) -> list[AggregatedItem]:
    """Load items from local sample JSON files."""

    movie_dir = sample_dir / "movie"
    episode_dir = sample_dir / "episode"

    movie_data = _read_json(movie_dir / "plex.json")["MediaContainer"]["Metadata"][0]
    imdb_movie = IMDbTitle.model_validate(_read_json(movie_dir / "imdb.json"))
    tmdb_movie = TMDBMovie.model_validate(_read_json(movie_dir / "tmdb.json"))

    episode_data = _read_json(episode_dir / "plex.tv.json")["MediaContainer"][
        "Metadata"
    ][0]
    imdb_episode = IMDbTitle.model_validate(_read_json(episode_dir / "imdb.tv.json"))
    tmdb_show = TMDBShow.model_validate(_read_json(episode_dir / "tmdb.tv.json"))

    return [
        AggregatedItem(
            plex=_load_plex_movie(movie_data),
            imdb=imdb_movie,
            tmdb=tmdb_movie,
        ),
        AggregatedItem(
            plex=_load_plex_episode(episode_data),
            imdb=imdb_episode,
            tmdb=tmdb_show,
        ),
    ]


__all__ = ["_load_from_sample"]
