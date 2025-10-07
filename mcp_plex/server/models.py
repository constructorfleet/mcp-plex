"""Typed models shared across the Plex server package."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from plexapi.client import PlexClient


class PlexTag(TypedDict, total=False):
    """Representation of a Plex tag entry (actor, director, etc.)."""

    tag: NotRequired[str]
    name: NotRequired[str]


PersonEntry = str | PlexTag


class ExternalIds(TypedDict, total=False):
    """External identifier payload for indexed media."""

    id: NotRequired[str | int | None]


class PlexMediaMetadata(TypedDict, total=False):
    """Subset of Plex metadata stored in Qdrant payloads."""

    rating_key: NotRequired[str]
    guid: NotRequired[str]
    title: NotRequired[str]
    type: NotRequired[str]
    thumb: NotRequired[str]
    art: NotRequired[str]
    summary: NotRequired[str]
    tagline: NotRequired[str | list[str]]
    added_at: NotRequired[int]
    year: NotRequired[int]
    directors: NotRequired[list[PersonEntry]]
    writers: NotRequired[list[PersonEntry]]
    actors: NotRequired[list[PersonEntry]]
    grandparent_title: NotRequired[str]
    parent_title: NotRequired[str]
    index: NotRequired[int]
    parent_index: NotRequired[int]
    grandparent_thumb: NotRequired[str]
    original_title: NotRequired[str]


class AggregatedMediaItem(TypedDict, total=False):
    """Flattened media payload combining Plex and external data."""

    title: NotRequired[str]
    summary: NotRequired[str]
    type: NotRequired[str]
    year: NotRequired[int]
    added_at: NotRequired[int]
    show_title: NotRequired[str]
    season_number: NotRequired[int]
    episode_number: NotRequired[int]
    tagline: NotRequired[str | list[str]]
    reviews: NotRequired[list[str]]
    overview: NotRequired[str]
    plot: NotRequired[str]
    genres: NotRequired[list[str]]
    collections: NotRequired[list[str]]
    actors: NotRequired[list[PersonEntry]]
    directors: NotRequired[list[PersonEntry]]
    writers: NotRequired[list[PersonEntry]]
    imdb: NotRequired[ExternalIds]
    tmdb: NotRequired[ExternalIds]
    tvdb: NotRequired[ExternalIds]
    plex: NotRequired[PlexMediaMetadata]


class MediaSummaryIdentifiers(TypedDict, total=False):
    """Identifiers that help reference a summarized media item."""

    rating_key: NotRequired[str]
    imdb: NotRequired[str]
    tmdb: NotRequired[str]


class SummarizedMediaItem(TypedDict, total=False):
    """Concise description of a media item for LLM consumption."""

    title: NotRequired[str]
    type: NotRequired[str]
    year: NotRequired[int]
    description: NotRequired[str]
    genres: NotRequired[list[str]]
    collections: NotRequired[list[str]]
    actors: NotRequired[list[str]]
    directors: NotRequired[list[str]]
    writers: NotRequired[list[str]]
    show: NotRequired[str]
    season: NotRequired[int]
    episode: NotRequired[int]
    identifiers: NotRequired[MediaSummaryIdentifiers]


class MediaSummaryResponse(TypedDict):
    """Container for summarized media items."""

    total_results: int
    results: list[SummarizedMediaItem]


class QdrantMediaPayload(TypedDict, total=False):
    """Raw payload stored within Qdrant records."""

    data: NotRequired[AggregatedMediaItem]
    title: NotRequired[str]
    summary: NotRequired[str]
    type: NotRequired[str]
    year: NotRequired[int]
    added_at: NotRequired[int]
    show_title: NotRequired[str]
    season_number: NotRequired[int]
    episode_number: NotRequired[int]
    tagline: NotRequired[str | list[str]]
    reviews: NotRequired[list[str]]
    overview: NotRequired[str]
    plot: NotRequired[str]
    genres: NotRequired[list[str]]
    collections: NotRequired[list[str]]
    actors: NotRequired[list[PersonEntry]]
    directors: NotRequired[list[PersonEntry]]
    writers: NotRequired[list[PersonEntry]]
    imdb: NotRequired[ExternalIds]
    tmdb: NotRequired[ExternalIds]
    tvdb: NotRequired[ExternalIds]
    plex: NotRequired[PlexMediaMetadata]


class PlexPlayerMetadata(TypedDict, total=False):
    """Metadata describing a Plex player that can receive playback commands."""

    name: NotRequired[str]
    product: NotRequired[str]
    display_name: str
    friendly_names: list[str]
    machine_identifier: NotRequired[str]
    client_identifier: NotRequired[str]
    address: NotRequired[str]
    port: NotRequired[int]
    provides: set[str]
    client: NotRequired[PlexClient | None]


__all__ = [
    "PlexTag",
    "PersonEntry",
    "ExternalIds",
    "PlexMediaMetadata",
    "AggregatedMediaItem",
    "MediaSummaryIdentifiers",
    "SummarizedMediaItem",
    "MediaSummaryResponse",
    "QdrantMediaPayload",
    "PlexPlayerMetadata",
]
