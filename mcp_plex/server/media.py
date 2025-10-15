"""Media helper functions for Plex server tools."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TYPE_CHECKING, cast

from qdrant_client import models

from ..common.types import JSONValue
from .models import (
    AggregatedMediaItem,
    MediaSummaryIdentifiers,
    MediaSummaryResponse,
    PlexMediaMetadata,
    QdrantMediaPayload,
    SummarizedMediaItem,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ..common.cache import MediaCache
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


def _collect_cache_keys(
    data: AggregatedMediaItem,
    plex_info: PlexMediaMetadata,
    *extra_identifiers: str | int | float | None,
) -> set[str]:
    """Gather normalized cache keys from an aggregated media payload."""

    cache_keys: set[str] = set()
    for candidate in extra_identifiers:
        normalized = _normalize_identifier(candidate)
        if normalized:
            cache_keys.add(normalized)

    rating_key = _normalize_identifier(plex_info.get("rating_key"))
    if rating_key:
        cache_keys.add(rating_key)

    guid = _normalize_identifier(plex_info.get("guid"))
    if guid:
        cache_keys.add(guid)

    for source_key in ("imdb", "tmdb", "tvdb"):
        source_value = data.get(source_key)
        if isinstance(source_value, Mapping):
            source_id = _normalize_identifier(source_value.get("id"))
            if source_id:
                cache_keys.add(source_id)

    return cache_keys


def _cache_media_artwork(
    cache: "MediaCache", cache_keys: set[str], plex_info: PlexMediaMetadata
) -> None:
    """Persist poster and background URLs for each cache key."""

    if not cache_keys:
        return

    thumb = plex_info.get("thumb")
    if isinstance(thumb, str) and thumb:
        for cache_key in cache_keys:
            cache.set_poster(cache_key, thumb)

    art = plex_info.get("art")
    if isinstance(art, str) and art:
        for cache_key in cache_keys:
            cache.set_background(cache_key, art)


def _ensure_rating_key_cached(
    cache_keys: set[str], plex_info: PlexMediaMetadata
) -> set[str]:
    """Guarantee the Plex rating key is part of the cache key set."""

    rating_key = _normalize_identifier(plex_info.get("rating_key"))
    if rating_key:
        cache_keys.add(rating_key)
    return cache_keys


def _identifier_matches_payload(
    identifier: str | int | float | None, payload: AggregatedMediaItem
) -> bool:
    """Return True when an identifier corresponds to cached payload fields."""

    normalized_identifier = _normalize_identifier(identifier)
    if not normalized_identifier:
        return False
    plex_info = _extract_plex_metadata(payload)
    cache_keys = _collect_cache_keys(payload, plex_info)
    return normalized_identifier in cache_keys


def _get_cached_payload(
    cache: "MediaCache", identifier: str | int | float | None
) -> AggregatedMediaItem | None:
    """Return a cached payload when the identifier matches stored keys."""

    normalized_identifier = _normalize_identifier(identifier)
    cache_keys_to_try: list[str] = []
    if normalized_identifier:
        cache_keys_to_try.append(normalized_identifier)
    if isinstance(identifier, str) and normalized_identifier != identifier:
        cache_keys_to_try.append(identifier)
    for cache_key in cache_keys_to_try:
        cached = cache.get_payload(cache_key)
        if cached is None:
            continue
        candidate = cast(AggregatedMediaItem, cached)
        if _identifier_matches_payload(identifier, candidate):
            return candidate
    return None


async def _get_media_data(server: "PlexServer", identifier: str) -> AggregatedMediaItem:
    """Return the first matching media record's payload."""

    cached_payload = _get_cached_payload(server.cache, identifier)
    if cached_payload is not None:
        return cached_payload
    records = await _find_records(server, identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    payload = _flatten_payload(cast(Mapping[str, JSONValue] | None, records[0].payload))
    data = payload

    plex_data = _extract_plex_metadata(data)
    cache_keys = _collect_cache_keys(data, plex_data, identifier)
    cache_keys = _ensure_rating_key_cached(cache_keys, plex_data)

    for cache_key in cache_keys:
        server.cache.set_payload(cache_key, cast(dict[str, JSONValue], payload))

    _cache_media_artwork(server.cache, cache_keys, plex_data)
    return payload


def _coerce_text(value: Any) -> str | None:
    """Convert raw values into a cleaned string when possible."""

    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [
            str(item).strip()
            for item in value
            if isinstance(item, str) and item.strip()
        ]
        if parts:
            return "; ".join(parts)
    return None


def _coerce_string_list(value: Any) -> list[str]:
    """Extract a list of readable strings from mixed payload values."""

    items: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            items.append(text)
        return items
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for entry in value:
            if isinstance(entry, str):
                text = entry.strip()
                if text:
                    items.append(text)
            elif isinstance(entry, Mapping):
                for key in ("tag", "name", "title", "role"):
                    maybe = entry.get(key)
                    if isinstance(maybe, str):
                        text = maybe.strip()
                        if text:
                            items.append(text)
                            break
    return items


def _first_text(*values: Any) -> str | None:
    """Return the first non-empty string from the provided values."""

    for value in values:
        text = _coerce_text(value)
        if text:
            return text
    return None


def _extract_review_snippet(value: Any) -> str | None:
    """Return a representative review snippet if one exists."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    for entry in value:
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                return text
        elif isinstance(entry, Mapping):
            for key in ("quote", "summary", "text", "content"):
                maybe = entry.get(key)
                if isinstance(maybe, str):
                    text = maybe.strip()
                    if text:
                        return text
    return None


def summarize_media_items_for_llm(
    items: Sequence[AggregatedMediaItem],
) -> MediaSummaryResponse:
    """Create a concise summary that can be read by a voice assistant or LLM."""

    summaries: list[SummarizedMediaItem] = []
    for media in items:
        plex_info = _extract_plex_metadata(media)
        summary_item: SummarizedMediaItem = {}

        title = _first_text(media.get("title"), plex_info.get("title"))
        if title:
            summary_item["title"] = title

        media_type = _first_text(media.get("type"), plex_info.get("type"))
        if media_type:
            summary_item["type"] = media_type

        year = media.get("year")
        if year is None:
            year = plex_info.get("year")
        if isinstance(year, int):
            summary_item["year"] = year

        show = _first_text(
            media.get("show_title"),
            plex_info.get("grandparent_title"),
            plex_info.get("parent_title"),
        )
        if show:
            summary_item["show"] = show

        season = media.get("season_number")
        if season is None:
            season = plex_info.get("parent_index")
        if isinstance(season, int):
            summary_item["season"] = season

        episode = media.get("episode_number")
        if episode is None:
            episode = plex_info.get("index")
        if isinstance(episode, int):
            summary_item["episode"] = episode

        genres = _coerce_string_list(media.get("genres"))
        if genres:
            summary_item["genres"] = genres

        collections = _coerce_string_list(media.get("collections"))
        if collections:
            summary_item["collections"] = collections

        actors = _coerce_string_list(media.get("actors"))
        if not actors:
            actors = _coerce_string_list(plex_info.get("actors"))
        if actors:
            summary_item["actors"] = actors

        directors = _coerce_string_list(media.get("directors"))
        if not directors:
            directors = _coerce_string_list(plex_info.get("directors"))
        if directors:
            summary_item["directors"] = directors

        writers = _coerce_string_list(media.get("writers"))
        if not writers:
            writers = _coerce_string_list(plex_info.get("writers"))
        if writers:
            summary_item["writers"] = writers

        tagline = _first_text(media.get("tagline"), plex_info.get("tagline"))
        main_summary = _first_text(
            media.get("summary"),
            plex_info.get("summary"),
            media.get("overview"),
            media.get("plot"),
        )
        review_snippet = _extract_review_snippet(media.get("reviews"))

        description_parts: list[str] = []
        if tagline:
            description_parts.append(tagline)
        if main_summary:
            description_parts.append(main_summary)
        elif review_snippet:
            description_parts.append(review_snippet)
        if actors:
            description_parts.append(f"Starring {', '.join(actors[:5])}")
        if show and summary_item.get("type") == "episode":
            description_parts.append(f"Episode of {show}")

        description = " ".join(description_parts)
        if description:
            summary_item["description"] = description

        identifiers: MediaSummaryIdentifiers = {}
        rating_key = _normalize_identifier(plex_info.get("rating_key"))
        if rating_key:
            identifiers["rating_key"] = rating_key
        imdb_value = media.get("imdb")
        if isinstance(imdb_value, Mapping):
            imdb_id = _normalize_identifier(imdb_value.get("id"))
            if imdb_id:
                identifiers["imdb"] = imdb_id
        tmdb_value = media.get("tmdb")
        if isinstance(tmdb_value, Mapping):
            tmdb_id = _normalize_identifier(tmdb_value.get("id"))
            if tmdb_id:
                identifiers["tmdb"] = tmdb_id
        if identifiers:
            summary_item["identifiers"] = identifiers

        summaries.append(summary_item)

    return {
        "total_results": len(items),
        "results": summaries,
    }
