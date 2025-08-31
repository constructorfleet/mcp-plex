"""Type definitions for Plex metadata and external services."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


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
    id: int
    name: str
    overview: Optional[str] = None
    season_number: Optional[int] = None
    episode_number: Optional[int] = None
    air_date: Optional[str] = None


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
    type: Literal["movie", "episode"]
    title: str
    summary: Optional[str] = None
    year: Optional[int] = None
    guids: List[PlexGuid] = Field(default_factory=list)
    thumb: Optional[str] = None
    art: Optional[str] = None
    tagline: Optional[str] = None
    content_rating: Optional[str] = None
    directors: List[PlexPerson] = Field(default_factory=list)
    writers: List[PlexPerson] = Field(default_factory=list)
    actors: List[PlexPerson] = Field(default_factory=list)


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
    "TMDBShow",
    "TMDBEpisode",
    "TMDBItem",
    "PlexGuid",
    "PlexPerson",
    "PlexItem",
    "AggregatedItem",
    "ExternalIDs",
]
