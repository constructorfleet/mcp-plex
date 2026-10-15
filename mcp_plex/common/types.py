"""Type definitions for Plex metadata and external services."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, Field, model_validator


class IMDbRating(BaseModel):
    """Subset of IMDb rating information."""

    aggregateRating: float | None = None
    voteCount: int | None = None


class IMDbImage(BaseModel):
    """Simplified representation of an IMDb image."""

    url: str


class IMDbName(BaseModel):
    """Minimal IMDb name record."""

    id: str
    displayName: str


class IMDbTitle(BaseModel):
    """Subset of an IMDb title record with people and artwork."""

    id: str
    type: str
    primaryTitle: str
    startYear: int | None = None
    runtimeSeconds: int | None = None
    genres: list[str] = Field(default_factory=list)
    rating: IMDbRating | None = None
    plot: str | None = None
    primaryImage: IMDbImage | None = None
    directors: list[IMDbName] = Field(default_factory=list)
    writers: list[IMDbName] = Field(default_factory=list)
    stars: list[IMDbName] = Field(default_factory=list)


class TMDBGenre(BaseModel):
    id: int
    name: str


class TMDBSeason(BaseModel):
    season_number: int
    name: str
    air_date: str | None = None


class TMDBCrewMember(BaseModel):
    id: int | None = None
    name: str | None = None
    original_name: str | None = None
    known_for_department: str | None = None
    department: str | None = None
    job: str | None = None
    credit_id: str | None = None
    adult: bool | None = None
    gender: int | None = None
    popularity: float | None = None
    profile_path: str | None = None


class TMDBGuestStar(BaseModel):
    id: int | None = None
    name: str | None = None
    original_name: str | None = None
    known_for_department: str | None = None
    character: str | None = None
    credit_id: str | None = None
    order: int | None = None
    adult: bool | None = None
    gender: int | None = None
    popularity: float | None = None
    profile_path: str | None = None


class TMDBMovie(BaseModel):
    id: int
    title: str
    overview: str | None = None
    release_date: str | None = None
    imdb_id: str | None = None
    genres: list[TMDBGenre] = Field(default_factory=list)
    poster_path: str | None = None
    backdrop_path: str | None = None
    tagline: str | None = None
    reviews: list[dict] = Field(default_factory=list)

    @classmethod
    def model_validate(cls, data):  # type: ignore[override]
        if isinstance(data, dict) and isinstance(data.get("reviews"), dict):
            data = data.copy()
            data["reviews"] = data["reviews"].get("results", [])
        return super().model_validate(data)


class TMDBShow(BaseModel):
    id: int
    name: str
    overview: str | None = None
    first_air_date: str | None = None
    last_air_date: str | None = None
    genres: list[TMDBGenre] = Field(default_factory=list)
    seasons: list[TMDBSeason] = Field(default_factory=list)
    poster_path: str | None = None
    backdrop_path: str | None = None
    tagline: str | None = None
    reviews: list[dict] = Field(default_factory=list)

    @classmethod
    def model_validate(cls, data):  # type: ignore[override]
        if isinstance(data, dict) and isinstance(data.get("reviews"), dict):
            data = data.copy()
            data["reviews"] = data["reviews"].get("results", [])
        return super().model_validate(data)


class TMDBEpisode(BaseModel):
    id: str
    name: str
    overview: str | None = None
    season_number: int | None = None
    episode_number: int | None = None
    air_date: str | None = None
    episode_type: str | None = None
    production_code: str | None = None
    runtime: int | None = None
    show_id: int | None = None
    still_path: str | None = None
    vote_average: float | None = None
    vote_count: int | None = None
    crew: list[TMDBCrewMember] = Field(default_factory=list)
    guest_stars: list[TMDBGuestStar] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalise_episode_id(cls, data):
        """Ensure episode IDs are populated when missing from TMDb payloads."""

        if isinstance(data, dict):
            payload = data.copy()
            episode_id = payload.get("id")
            if episode_id is not None:
                payload["id"] = str(episode_id)
            else:
                show_id = payload.get("show_id")
                season_number = payload.get("season_number")
                episode_number = payload.get("episode_number")
                if (
                    show_id is not None
                    and season_number is not None
                    and episode_number is not None
                ):
                    payload["id"] = (
                        f"{show_id}/season/{season_number}/episode/{episode_number}"
                    )
            return payload
        return data


TMDBItem = TMDBMovie | TMDBShow | TMDBEpisode


class PlexGuid(BaseModel):
    id: str


class PlexPerson(BaseModel):
    """Representation of a person in Plex metadata."""

    id: int
    tag: str
    thumb: str | None = None
    role: str | None = None


class PlexItem(BaseModel):
    rating_key: str
    guid: str
    type: Literal["movie", "episode", "show", "season"]
    title: str
    show_title: str | None = None
    season_title: str | None = None
    season_number: int | None = None
    episode_number: int | None = None
    summary: str | None = None
    year: int | None = None
    added_at: datetime | None = None
    guids: list[PlexGuid] = Field(default_factory=list)
    thumb: str | None = None
    art: str | None = None
    tagline: str | None = None
    content_rating: str | None = None
    directors: list[PlexPerson] = Field(default_factory=list)
    writers: list[PlexPerson] = Field(default_factory=list)
    actors: list[PlexPerson] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)


class AggregatedItem(BaseModel):
    """Aggregated Plex/IMDb/TMDb metadata."""

    plex: PlexItem
    imdb: IMDbTitle | None = None
    tmdb: TMDBItem | None = None


@dataclass
class ExternalIDs:
    imdb: str | None = None
    tmdb: str | None = None


__all__ = [
    "AggregatedItem",
    "ExternalIDs",
    "IMDbImage",
    "IMDbName",
    "IMDbRating",
    "IMDbTitle",
    "PlexGuid",
    "PlexItem",
    "PlexPerson",
    "TMDBCrewMember",
    "TMDBEpisode",
    "TMDBGenre",
    "TMDBGuestStar",
    "TMDBItem",
    "TMDBMovie",
    "TMDBSeason",
    "TMDBShow",
]
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
JSONMapping: TypeAlias = Mapping[str, JSONValue]
MutableJSONMapping: TypeAlias = MutableMapping[str, JSONValue]
