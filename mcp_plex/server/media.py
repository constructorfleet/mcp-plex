"""Media helper functions for Plex server tools."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, MutableMapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

from qdrant_client import models
from rapidfuzz import fuzz

from ..common import strip_leading_article
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


def _is_imdb_identifier(identifier: str) -> bool:
    text = identifier.strip().lower()
    return bool(text.startswith("tt") and text[2:].isdigit())


def _should_use_identifier_filter(identifier: str) -> bool:
    return identifier.isdigit() or _is_imdb_identifier(identifier)


async def _rerank_records(
    server: "PlexServer",
    query_text: str,
    points: Sequence[models.Record | models.ScoredPoint],
    *,
    title: str | None = None,
) -> list[models.Record | models.ScoredPoint]:
    from mcp_plex.server.tools.media_library import _rerank_media_candidates

    flat_items = [
        _flatten_payload(cast(Mapping[str, JSONValue] | None, point.payload))
        for point in points
    ]
    reranked_items = await _rerank_media_candidates(
        server,
        query_text,
        flat_items,
        title=title or query_text,
    )
    rating_key_to_index: dict[str, int] = {}
    for idx, item in enumerate(flat_items):
        rating_key = _normalize_identifier(
            _extract_plex_metadata(item).get("rating_key")
        )
        if rating_key and rating_key not in rating_key_to_index:
            rating_key_to_index[rating_key] = idx

    ordered: list[models.Record | models.ScoredPoint] = []
    used_indexes: set[int] = set()
    for item in reranked_items:
        rating_key = _normalize_identifier(
            _extract_plex_metadata(item).get("rating_key")
        )
        if rating_key is None:
            continue
        idx = rating_key_to_index.get(rating_key)
        if idx is None or idx in used_indexes:
            continue
        ordered.append(points[idx])
        used_indexes.add(idx)

    if len(ordered) == len(points):
        return ordered

    for idx, point in enumerate(points):
        if idx in used_indexes:
            continue
        ordered.append(point)
    return ordered


async def _find_records(
    server: "PlexServer",
    identifier: str,
    limit: int = 5,
    *,
    allow_vector: bool = True,
    min_title_ratio: int | None = None,
) -> list[models.Record | models.ScoredPoint]:
    """Locate records matching an identifier or title."""

    normalized_identifier = identifier.strip()
    if not normalized_identifier:
        return []

    try:
        record_id: Any = (
            int(normalized_identifier)
            if normalized_identifier.isdigit()
            else normalized_identifier
        )
        recs = await server.qdrant_client.retrieve(
            "media-items", ids=[record_id], with_payload=True
        )
        if recs:
            return cast(list[models.Record | models.ScoredPoint], recs)
    except Exception:
        pass

    if _should_use_identifier_filter(normalized_identifier):
        should: list[models.FieldCondition] = [
            models.FieldCondition(
                key="data.plex.rating_key",
                match=models.MatchValue(value=normalized_identifier),
            ),
            models.FieldCondition(
                key="data.imdb.id",
                match=models.MatchValue(value=normalized_identifier),
            ),
        ]
        if normalized_identifier.isdigit():
            should.append(
                models.FieldCondition(
                    key="data.tmdb.id",
                    match=models.MatchValue(value=int(normalized_identifier)),
                )
            )
        scroll_filter = models.Filter(should=cast(list[models.Condition], should))
        raw_points, _ = await server.qdrant_client.scroll(
            collection_name="media-items",
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
        )
        scroll_points: list[models.Record | models.ScoredPoint] = cast(
            list[models.Record | models.ScoredPoint], raw_points
        )
        if scroll_points:
            if len(scroll_points) > 1:
                scroll_points = await _rerank_records(
                    server,
                    normalized_identifier,
                    scroll_points,
                    title=normalized_identifier,
                )
            return scroll_points
        if not allow_vector:
            return []


    if not allow_vector:
        return []

    search_variants: list[str] = []
    slug_source = strip_leading_article(normalized_identifier) or normalized_identifier
    for candidate in (normalized_identifier, slug_source):
        if candidate and candidate not in search_variants:
            search_variants.append(candidate)

    vector_queries: list[tuple[str, models.Document]] = []
    dense_model = getattr(server.settings, "dense_model", None)
    sparse_model = getattr(server.settings, "sparse_model", None)
    for text_value in search_variants:
        if dense_model:
            vector_queries.append(
                (
                    "dense",
                    models.Document(text=text_value, model=dense_model),
                )
            )
        if sparse_model:
            vector_queries.append(
                (
                    "sparse",
                    models.Document(text=text_value, model=sparse_model),
                )
            )

    if not vector_queries:
        return []

    candidate_limit = max(limit * 3, limit)
    prefetch_entries = [
        models.Prefetch(
            query=models.NearestQuery(nearest=document),
            using=name,
            limit=candidate_limit,
        )
        for name, document in vector_queries
    ]

    if len(prefetch_entries) == 1:
        query_obj = cast(models.Query, prefetch_entries[0].query)
        using_param: str | None = prefetch_entries[0].using
        prefetch_param: Sequence[models.Prefetch] | None = None
    else:
        query_obj = models.FusionQuery(fusion=models.Fusion.RRF)
        using_param = None
        prefetch_param = prefetch_entries

    res = await server.qdrant_client.query_points(
        collection_name="media-items",
        query=query_obj,
        using=using_param,
        prefetch=prefetch_param,
        limit=limit,
        with_payload=True,
    )
    points: list[models.Record | models.ScoredPoint] = list(res.points or [])
    if len(points) > 1:
        points = await _rerank_records(
            server, normalized_identifier, points, title=normalized_identifier
        )

    if min_title_ratio is not None and points:
        filtered_points: list[models.Record | models.ScoredPoint] = []
        for point in points:
            payload = cast(Mapping[str, JSONValue] | None, point.payload)
            flat = _flatten_payload(payload)
            candidate_title = _first_text(
                flat.get("title"),
                _extract_plex_metadata(flat).get("title"),
            )
            if not candidate_title:
                continue
            ratio = fuzz.ratio(
                normalized_identifier.lower(), candidate_title.lower()
            )
            if ratio >= min_title_ratio:
                filtered_points.append(point)
        points = filtered_points

    return points


def _flatten_payload(payload: Mapping[str, JSONValue] | None) -> AggregatedMediaItem:
    """Merge top-level payload fields with the nested data block."""

    data: dict[str, JSONValue] = {}
    if not payload:
        return cast(AggregatedMediaItem, data)
    payload_dict = cast(QdrantMediaPayload, payload)
    base = payload_dict.get("data")
    if isinstance(base, dict):
        data.update(cast(Mapping[str, JSONValue], base))
    for key, value in payload_dict.items():
        if key == "data":
            continue
        data[key] = cast(JSONValue, value)
    _ensure_epoch_added_at(data)
    return cast(AggregatedMediaItem, data)


def _ensure_epoch_added_at(data: MutableMapping[str, JSONValue]) -> None:
    """Coerce any ``added_at`` timestamps to Linux epoch seconds."""

    _update_epoch_field(data, "added_at")

    plex_value = data.get("plex")
    if isinstance(plex_value, MutableMapping):
        _update_epoch_field(plex_value, "added_at")
    elif isinstance(plex_value, Mapping):
        normalized = _normalize_epoch_timestamp(plex_value.get("added_at"))
        if normalized is not None:
            updated = dict(plex_value)
            updated["added_at"] = normalized
            data["plex"] = cast(JSONValue, updated)


def _update_epoch_field(
    container: MutableMapping[str, JSONValue], key: str
) -> bool:
    """Normalize a timestamp field in-place when possible."""

    normalized = _normalize_epoch_timestamp(container.get(key))
    if normalized is None:
        return False
    container[key] = normalized
    return True


def _normalize_epoch_timestamp(value: Any) -> int | None:
    """Convert mixed timestamp representations into epoch seconds."""

    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


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


def _normalize_history_rating_key(value: object) -> str | None:
    """Normalize mixed history identifiers into rating keys."""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    try:
        normalized = str(value).strip()
    except Exception:  # noqa: BLE001 - guard against unusual reprs
        return None
    return normalized or None


def _normalize_history_rating_keys(values: Iterable[object]) -> set[str]:
    """Return normalized rating keys derived from history values."""

    normalized: set[str] = set()
    for value in values:
        rating_key = _normalize_history_rating_key(value)
        if rating_key:
            normalized.add(rating_key)
    return normalized


def _rating_key_from_media(media: AggregatedMediaItem) -> str | None:
    """Extract a normalized rating key from an aggregated media payload."""

    plex_info = _extract_plex_metadata(media)
    return _normalize_history_rating_key(plex_info.get("rating_key"))


def _history_exclusion_filter(
    watched_keys: Collection[str],
) -> models.Filter | None:
    """Create a Qdrant filter that omits watched rating keys."""

    if not watched_keys:
        return None
    return models.Filter(
        must_not=[
            models.FieldCondition(
                key="data.plex.rating_key",
                match=models.MatchAny(any=list(watched_keys)),
            )
        ]
    )


def _filter_watched_recommendations(
    candidates: Iterable[AggregatedMediaItem],
    watched_keys: Collection[str],
    *,
    positive_rating_key: str | None = None,
    limit: int | None = None,
) -> list[AggregatedMediaItem]:
    """Filter recommendation candidates against watched history and limits."""

    filtered: list[AggregatedMediaItem] = []
    watched_lookup = set(watched_keys)
    for media in candidates:
        rating_key = _rating_key_from_media(media)
        if watched_lookup and rating_key and rating_key in watched_lookup:
            continue
        if positive_rating_key and rating_key == positive_rating_key:
            continue
        filtered.append(media)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


async def _history_recommendations(
    server: "PlexServer",
    rating_keys: Collection[str],
    limit: int,
) -> list[AggregatedMediaItem]:
    """Load aggregated media items corresponding to history rating keys."""

    normalized_keys = _normalize_history_rating_keys(rating_keys)
    if not normalized_keys or limit <= 0:
        return []

    candidate_limit = max(limit * 2, limit, len(normalized_keys))
    candidate_limit = min(candidate_limit, 100)
    filter_obj = models.Filter(
        must=[
            models.FieldCondition(
                key="data.plex.rating_key",
                match=models.MatchAny(any=list(normalized_keys)),
            )
        ]
    )
    points, _ = await server.qdrant_client.scroll(
        collection_name="media-items",
        scroll_filter=filter_obj,
        limit=candidate_limit,
        with_payload=True,
    )
    results: list[AggregatedMediaItem] = []
    for point in points:
        data = _flatten_payload(cast(Mapping[str, JSONValue] | None, point.payload))
        rating_key = _rating_key_from_media(data)
        if rating_key and rating_key in normalized_keys:
            results.append(data)
        if len(results) >= limit:
            break
    return results


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


async def _get_media_data(
    server: "PlexServer", identifier: str, *, allow_vector: bool | None = None
) -> AggregatedMediaItem:
    """Return the first matching media record's payload."""

    cached_payload = _get_cached_payload(server.cache, identifier)
    if cached_payload is not None:
        return cached_payload

    identifier_text = identifier.strip()
    allow_vector_lookup = allow_vector
    if allow_vector_lookup is None:
        allow_vector_lookup = not (
            identifier_text and _should_use_identifier_filter(identifier_text)
        )

    records = await _find_records(server, identifier, limit=1, allow_vector=False)
    if not records and allow_vector_lookup:
        records = await _find_records(server, identifier, limit=1, allow_vector=True)
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
