"""Media helper functions for Plex server tools."""

from __future__ import annotations

from typing import Any, Mapping, TYPE_CHECKING, cast

from qdrant_client import models

from ..common.types import JSONValue
from .models import AggregatedMediaItem, PlexMediaMetadata, QdrantMediaPayload

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from . import PlexServer


async def _find_records(
    server: "PlexServer", identifier: str, limit: int = 5
) -> list[models.Record]:
    """Locate records matching an identifier or title."""

    try:
        record_id: Any = int(identifier) if identifier.isdigit() else identifier
        recs = await server.qdrant_client.retrieve(
            "media-items", ids=[record_id], with_payload=True
        )
        if recs:
            return recs
    except Exception:
        pass

    should = [
        models.FieldCondition(
            key="data.plex.rating_key", match=models.MatchValue(value=identifier)
        ),
        models.FieldCondition(
            key="data.imdb.id", match=models.MatchValue(value=identifier)
        ),
    ]
    if identifier.isdigit():
        should.append(
            models.FieldCondition(
                key="data.tmdb.id", match=models.MatchValue(value=int(identifier))
            )
        )
    should.append(
        models.FieldCondition(key="title", match=models.MatchText(text=identifier))
    )
    flt = models.Filter(should=should)
    points, _ = await server.qdrant_client.scroll(
        collection_name="media-items",
        limit=limit,
        scroll_filter=flt,
        with_payload=True,
    )
    return points


def _flatten_payload(payload: Mapping[str, JSONValue] | None) -> AggregatedMediaItem:
    """Merge top-level payload fields with the nested data block."""

    data: dict[str, JSONValue] = {}
    if not payload:
        return cast(AggregatedMediaItem, data)
    payload_dict = cast(QdrantMediaPayload, payload)
    base = payload_dict.get("data")
    if isinstance(base, dict):
        data.update(base)
    for key, value in payload_dict.items():
        if key == "data":
            continue
        data[key] = value
    return cast(AggregatedMediaItem, data)


def _normalize_identifier(value: str | int | float | None) -> str | None:
    """Convert mixed identifier formats into a normalized string."""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    try:
        normalized_value = str(value)
    except Exception:
        return None
    return normalized_value.strip() or None


def _extract_plex_metadata(media: AggregatedMediaItem) -> PlexMediaMetadata:
    """Return Plex metadata block from an aggregated media item."""

    plex_value = media.get("plex")
    if isinstance(plex_value, dict):
        return cast(PlexMediaMetadata, plex_value)
    return cast(PlexMediaMetadata, {})


async def _get_media_data(server: "PlexServer", identifier: str) -> AggregatedMediaItem:
    """Return the first matching media record's payload."""

    cached = server.cache.get_payload(identifier)
    if cached is not None:
        return cast(AggregatedMediaItem, cached)
    records = await _find_records(server, identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    payload = _flatten_payload(cast(Mapping[str, JSONValue] | None, records[0].payload))
    data = payload

    cache_keys: set[str] = set()

    lookup_key = _normalize_identifier(identifier)
    if lookup_key:
        cache_keys.add(lookup_key)

    plex_data = _extract_plex_metadata(data)
    rating_key = _normalize_identifier(plex_data.get("rating_key"))
    if rating_key:
        cache_keys.add(rating_key)
    guid = _normalize_identifier(plex_data.get("guid"))
    if guid:
        cache_keys.add(guid)

    for source_key in ("imdb", "tmdb", "tvdb"):
        source_value = data.get(source_key)
        if isinstance(source_value, dict):
            source_id = _normalize_identifier(source_value.get("id"))
            if source_id:
                cache_keys.add(source_id)

    for cache_key in cache_keys:
        server.cache.set_payload(cache_key, cast(dict[str, JSONValue], payload))

    if rating_key:
        thumb = plex_data.get("thumb")
        if isinstance(thumb, str) and thumb:
            server.cache.set_poster(rating_key, thumb)
        art = plex_data.get("art")
        if isinstance(art, str) and art:
            server.cache.set_background(rating_key, art)
    return payload
