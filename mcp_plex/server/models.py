"""Typed models shared across the Plex server package."""

from __future__ import annotations

from collections.abc import ItemsView, Iterator, KeysView, Mapping, ValuesView
from typing import NotRequired, TypedDict

from plexapi.client import PlexClient
from pydantic import BaseModel, ConfigDict, Field


class _DictLikeModel(BaseModel, Mapping[str, object]):
    """Base model that preserves dict-like ergonomics for responses."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
        populate_by_name=True,
    )

    def _as_dict(self) -> dict[str, object]:
        return self.model_dump(mode="python")

    def __getitem__(self, key: str) -> object:
        return self._as_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._as_dict())

    def __len__(self) -> int:
        return len(self._as_dict())

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self._as_dict()

    def get(self, key: str, default: object | None = None) -> object | None:
        return self._as_dict().get(key, default)

    def items(self) -> ItemsView[str, object]:
        return self._as_dict().items()

    def keys(self) -> KeysView[str]:
        return self._as_dict().keys()

    def values(self) -> ValuesView[object]:
        return self._as_dict().values()


class PlexTag(TypedDict, total=False):
    """Representation of a Plex tag entry (actor, director, etc.)."""

    tag: NotRequired[str]
    name: NotRequired[str]


PersonEntry = str | PlexTag


class PlexTagModel(_DictLikeModel):
    """Pydantic representation of a Plex tag entry."""

    tag: str | None = None
    name: str | None = None


class ExternalIds(TypedDict, total=False):
    """External identifier payload for indexed media."""

    id: NotRequired[str | int | None]


class ExternalIdsModel(_DictLikeModel):
    """Pydantic representation of an external identifier payload."""

    id: str | int | None = None


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


class PlexMediaMetadataModel(_DictLikeModel):
    """Pydantic representation of Plex metadata stored in payloads."""

    rating_key: str | None = None
    guid: str | None = None
    title: str | None = None
    type: str | None = None
    thumb: str | None = None
    art: str | None = None
    summary: str | None = None
    tagline: str | list[str] | None = None
    added_at: int | None = None
    year: int | None = None
    directors: list[str | PlexTagModel] | None = None
    writers: list[str | PlexTagModel] | None = None
    actors: list[str | PlexTagModel] | None = None
    grandparent_title: str | None = None
    parent_title: str | None = None
    index: int | None = None
    parent_index: int | None = None
    grandparent_thumb: str | None = None
    original_title: str | None = None


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


class AggregatedMediaItemModel(_DictLikeModel):
    """Pydantic representation of an aggregated media payload."""

    title: str | None = None
    summary: str | None = None
    type: str | None = None
    year: int | None = None
    added_at: int | None = None
    show_title: str | None = None
    season_number: int | None = None
    episode_number: int | None = None
    tagline: str | list[str] | None = None
    reviews: list[str] | None = None
    overview: str | None = None
    plot: str | None = None
    genres: list[str] | None = None
    collections: list[str] | None = None
    actors: list[str | PlexTagModel] | None = None
    directors: list[str | PlexTagModel] | None = None
    writers: list[str | PlexTagModel] | None = None
    imdb: ExternalIdsModel | None = None
    tmdb: ExternalIdsModel | None = None
    tvdb: ExternalIdsModel | None = None
    plex: PlexMediaMetadataModel | None = None


class MediaSummaryIdentifiers(TypedDict, total=False):
    """Identifiers that help reference a summarized media item."""

    rating_key: NotRequired[str]
    imdb: NotRequired[str]
    tmdb: NotRequired[str]


class MediaSummaryIdentifiersModel(_DictLikeModel):
    """Pydantic representation of media summary identifiers."""

    rating_key: str | None = None
    imdb: str | None = None
    tmdb: str | int | None = None


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


class SummarizedMediaItemModel(_DictLikeModel):
    """Pydantic representation of a summarized media item."""

    title: str | None = None
    type: str | None = None
    year: int | None = None
    description: str | None = None
    genres: list[str] | None = None
    collections: list[str] | None = None
    actors: list[str] | None = None
    directors: list[str] | None = None
    writers: list[str] | None = None
    show: str | None = None
    season: int | None = None
    episode: int | None = None
    identifiers: MediaSummaryIdentifiersModel | None = None


class MediaSummaryResponse(TypedDict):
    """Container for summarized media items."""

    total_results: int
    results: list[SummarizedMediaItem]


class MediaSummaryResponseModel(_DictLikeModel):
    """Pydantic response wrapper for summarized media items."""

    total_results: int = 0
    results: list[SummarizedMediaItemModel] = Field(default_factory=list)


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


class PlayMediaResponseModel(_DictLikeModel):
    """Response payload returned when initiating playback on a player."""

    player: str | None = None
    rating_key: str | None = None
    title: str | None = None
    offset_seconds: int = 0
    player_capabilities: list[str] = Field(default_factory=list)


class QueueMediaResponseModel(_DictLikeModel):
    """Response payload returned when queueing media on a player."""

    player: str | None = None
    rating_key: str | None = None
    title: str | None = None
    position: str | None = None
    queue_size: int | None = None
    queue_version: int | None = None


class PlayerCommandResponseModel(_DictLikeModel):
    """Standardized response for player control commands."""

    player: str | None = None
    command: str
    media_type: str | None = None
    player_capabilities: list[str] = Field(default_factory=list)
    success: bool = True
    error: str | None = None
    audio_language: str | None = None
    audio_channels: int | None = None
    audio_stream_id: int | None = None
    subtitle_language: str | None = None
    subtitle_stream_id: int | None = None

    def with_updates(
        self, **updates: object | None
    ) -> "PlayerCommandResponseModel":
        """Return a copy of the response with additional fields set."""

        return self.model_copy(update=updates)


__all__ = [
    "PlexTag",
    "PlexTagModel",
    "PersonEntry",
    "ExternalIds",
    "ExternalIdsModel",
    "PlexMediaMetadata",
    "PlexMediaMetadataModel",
    "AggregatedMediaItem",
    "AggregatedMediaItemModel",
    "MediaSummaryIdentifiers",
    "MediaSummaryIdentifiersModel",
    "SummarizedMediaItem",
    "SummarizedMediaItemModel",
    "MediaSummaryResponse",
    "MediaSummaryResponseModel",
    "QdrantMediaPayload",
    "PlexPlayerMetadata",
    "PlayMediaResponseModel",
    "QueueMediaResponseModel",
    "PlayerCommandResponseModel",
]
