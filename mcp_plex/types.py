"""Type definitions for Plex metadata and external services."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Literal

from pydantic import BaseModel, Field


class IMDbRating(BaseModel):
    """Subset of IMDb rating information."""

    aggregateRating: Optional[float] = None
    voteCount: Optional[int] = None


class IMDbTitle(BaseModel):
    """Subset of an IMDb title record."""

    id: str
    type: str
    primaryTitle: str
    startYear: Optional[int] = None
    runtimeSeconds: Optional[int] = None
    genres: List[str] = Field(default_factory=list)
    rating: Optional[IMDbRating] = None
    plot: Optional[str] = None


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


class TMDBShow(BaseModel):
    id: int
    name: str
    overview: Optional[str] = None
    first_air_date: Optional[str] = None
    last_air_date: Optional[str] = None
    genres: List[TMDBGenre] = Field(default_factory=list)


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


class PlexItem(BaseModel):
    rating_key: str
    guid: str
    type: Literal["movie", "episode"]
    title: str
    summary: Optional[str] = None
    year: Optional[int] = None
    guids: List[PlexGuid] = Field(default_factory=list)


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
    "TMDBGenre",
    "TMDBMovie",
    "TMDBShow",
    "TMDBEpisode",
    "TMDBItem",
    "PlexGuid",
    "PlexItem",
    "AggregatedItem",
    "ExternalIDs",
]
