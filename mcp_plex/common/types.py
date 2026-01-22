"""Type definitions for Plex metadata and external services."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Mapping, MutableMapping, Optional, Sequence, TypeAlias

from pydantic import BaseModel, Field, model_validator


class IMDbRating(BaseModel):
    """Subset of IMDb rating information."""

    aggregateRating: Optional[float] = None
    voteCount: Optional[int] = None


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
    startYear: Optional[int] = None
    runtimeSeconds: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    rating: Optional[IMDbRating] = None
    plot: Optional[str] = None
    primaryImage: Optional[IMDbImage] = None
    directors: List[IMDbName] = Field(default_factory=list)
    writers: List[IMDbName] = Field(default_factory=list)
    stars: List[IMDbName] = Field(default_factory=list)


class TMDBGenre(BaseModel):
    id: int
    name: str


class TMDBSeason(BaseModel):
    season_number: int
    name: str
    air_date: Optional[str] = None


class TMDBCrewMember(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    original_name: Optional[str] = None
    known_for_department: Optional[str] = None
    department: Optional[str] = None
    job: Optional[str] = None
    credit_id: Optional[str] = None
    adult: Optional[bool] = None
    gender: Optional[int] = None
    popularity: Optional[float] = None
    profile_path: Optional[str] = None


class TMDBGuestStar(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    original_name: Optional[str] = None
    known_for_department: Optional[str] = None
    character: Optional[str] = None
    credit_id: Optional[str] = None
    order: Optional[int] = None
    adult: Optional[bool] = None
    gender: Optional[int] = None
    popularity: Optional[float] = None
    profile_path: Optional[str] = None


class TMDBMovie(BaseModel):
    id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    imdb_id: Optional[str] = None
    genres: List[TMDBGenre] = Field(default_factory=list)
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    tagline: Optional[str] = None
    reviews: List[dict] = Field(default_factory=list)

    @classmethod
    def model_validate(cls, data):  # type: ignore[override]
        if isinstance(data, dict) and isinstance(data.get("reviews"), dict):
            data = data.copy()
            data["reviews"] = data["reviews"].get("results", [])
        return super().model_validate(data)


class TMDBShow(BaseModel):
    id: int
    name: str
    overview: Optional[str] = None
    first_air_date: Optional[str] = None
    last_air_date: Optional[str] = None
    genres: List[TMDBGenre] = Field(default_factory=list)
    seasons: List[TMDBSeason] = Field(default_factory=list)
    poster_path: Optional[str] = None
    backdrop_path: Optional[str] = None
    tagline: Optional[str] = None
    reviews: List[dict] = Field(default_factory=list)

    @classmethod
    def model_validate(cls, data):  # type: ignore[override]
        if isinstance(data, dict) and isinstance(data.get("reviews"), dict):
            data = data.copy()
            data["reviews"] = data["reviews"].get("results", [])
        return super().model_validate(data)


class TMDBEpisode(BaseModel):
    id: str
    name: str
    overview: Optional[str] = None
    season_number: Optional[int] = None
    episode_number: Optional[int] = None
    air_date: Optional[str] = None
    episode_type: Optional[str] = None
    production_code: Optional[str] = None
    runtime: Optional[int] = None
    show_id: Optional[int] = None
    still_path: Optional[str] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    crew: List[TMDBCrewMember] = Field(default_factory=list)
    guest_stars: List[TMDBGuestStar] = Field(default_factory=list)

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
    thumb: Optional[str] = None
    role: Optional[str] = None


class PlexItem(BaseModel):
    rating_key: str
    guid: str
    type: Literal["movie", "episode", "show", "season"]
    title: str
    show_title: Optional[str] = None
    season_title: Optional[str] = None
    season_number: Optional[int] = None
    episode_number: Optional[int] = None
    summary: Optional[str] = None
    year: Optional[int] = None
    added_at: Optional[datetime] = None
    guids: List[PlexGuid] = Field(default_factory=list)
    thumb: Optional[str] = None
    art: Optional[str] = None
    tagline: Optional[str] = None
    content_rating: Optional[str] = None
    directors: List[PlexPerson] = Field(default_factory=list)
    writers: List[PlexPerson] = Field(default_factory=list)
    actors: List[PlexPerson] = Field(default_factory=list)
    genres: List[str] = Field(default_factory=list)
    collections: List[str] = Field(default_factory=list)


class AggregatedItem(BaseModel):
    """Aggregated Plex/IMDb/TMDb metadata."""

    plex: PlexItem
    imdb: Optional[IMDbTitle] = None
    tmdb: Optional[TMDBItem] = None


@dataclass
class ExternalIDs:
    imdb: Optional[str] = None
    tmdb: Optional[str] = None


__all__ = [
    "IMDbRating",
    "IMDbTitle",
    "IMDbImage",
    "IMDbName",
    "TMDBGenre",
    "TMDBMovie",
    "TMDBSeason",
    "TMDBShow",
    "TMDBEpisode",
    "TMDBCrewMember",
    "TMDBGuestStar",
    "TMDBItem",
    "PlexGuid",
    "PlexPerson",
    "PlexItem",
    "AggregatedItem",
    "ExternalIDs",
]
JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Sequence["JSONValue"] | Mapping[str, "JSONValue"]
JSONMapping: TypeAlias = Mapping[str, JSONValue]
MutableJSONMapping: TypeAlias = MutableMapping[str, JSONValue]
