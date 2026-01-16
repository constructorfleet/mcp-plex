"""Qdrant helper utilities shared across the loader pipeline."""

from __future__ import annotations

import asyncio
import math
import random
import time
import logging
import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, TypedDict

from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import WriteOrdering

from ..common.text import slugify, strip_leading_article
from ..common.types import AggregatedItem, JSONValue

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from . import QdrantRuntimeConfig

logger = logging.getLogger("mcp_plex.loader.qdrant")


def _is_local_qdrant(client: AsyncQdrantClient) -> bool:
    """Return ``True`` if *client* targets an in-process Qdrant instance."""

    inner = getattr(client, "_client", None)
    return bool(inner) and inner.__class__.__module__.startswith("qdrant_client.local")


# Known Qdrant-managed dense embedding models with their dimensionality and
# similarity metric. To support a new server-side embedding model, add an entry
# here with the appropriate vector size and `models.Distance` value.
_DENSE_MODEL_PARAMS: dict[str, tuple[int, models.Distance]] = {
    "BAAI/bge-small-en-v1.5": (384, models.Distance.COSINE),
    "BAAI/bge-base-en-v1.5": (768, models.Distance.COSINE),
    "BAAI/bge-large-en-v1.5": (1024, models.Distance.COSINE),
    "text-embedding-3-small": (1536, models.Distance.COSINE),
    "text-embedding-3-large": (3072, models.Distance.COSINE),
}


def _resolve_dense_model_params(model_name: str) -> tuple[int, models.Distance]:
    """Look up Qdrant vector parameters for a known dense embedding model."""

    try:
        return _DENSE_MODEL_PARAMS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown dense embedding model '{model_name}'. Update _DENSE_MODEL_PARAMS with the model's size and distance."
        ) from exc


async def _ensure_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    dense_size: int,
    dense_distance: models.Distance,
) -> None:
    """Create the collection and payload indexes if they do not already exist."""

    created_collection = False
    if not await client.collection_exists(collection_name):
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(size=dense_size, distance=dense_distance)
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        created_collection = True

    if not created_collection:
        return

    suppress_payload_warning = _is_local_qdrant(client)

    async def _create_index(
        field_name: str,
        field_schema: models.PayloadSchemaType | models.TextIndexParams,
    ) -> None:
        if suppress_payload_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Payload indexes have no effect in the local Qdrant.*",
                    category=UserWarning,
                )
                await client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
        else:
            await client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

    text_index = models.TextIndexParams(
        type=models.PayloadSchemaType.TEXT,
        tokenizer=models.TokenizerType.WORD,
        min_token_len=2,
        lowercase=True,
    )
    await _create_index("title", text_index)
    await _create_index("title_slug", models.PayloadSchemaType.KEYWORD)
    await _create_index("type", models.PayloadSchemaType.KEYWORD)
    await _create_index("year", models.PayloadSchemaType.INTEGER)
    await _create_index("added_at", models.PayloadSchemaType.INTEGER)
    await _create_index("actors", models.PayloadSchemaType.KEYWORD)
    await _create_index("directors", models.PayloadSchemaType.KEYWORD)
    await _create_index("writers", models.PayloadSchemaType.KEYWORD)
    await _create_index("genres", models.PayloadSchemaType.KEYWORD)
    await _create_index("show_title", models.PayloadSchemaType.KEYWORD)
    await _create_index("show_title_slug", models.PayloadSchemaType.KEYWORD)
    await _create_index("season_number", models.PayloadSchemaType.INTEGER)
    await _create_index("episode_number", models.PayloadSchemaType.INTEGER)
    await _create_index("collections", models.PayloadSchemaType.KEYWORD)
    await _create_index("summary", text_index)
    await _create_index("overview", text_index)
    await _create_index("plot", text_index)
    await _create_index("tagline", text_index)
    await _create_index("reviews", text_index)
    await _create_index("data.plex.rating_key", models.PayloadSchemaType.KEYWORD)
    await _create_index("data.imdb.id", models.PayloadSchemaType.KEYWORD)
    await _create_index("data.tmdb.id", models.PayloadSchemaType.INTEGER)


def _format_primary_title(item: AggregatedItem) -> str:
    """Format the primary title text for ``item``."""

    primary_title = item.plex.title
    if item.plex.type == "episode":
        title_bits: list[str] = []
        if item.plex.show_title:
            title_bits.append(item.plex.show_title)
        se_parts: list[str] = []
        if item.plex.season_number is not None:
            se_parts.append(f"S{item.plex.season_number:02d}")
        if item.plex.episode_number is not None:
            se_parts.append(f"E{item.plex.episode_number:02d}")
        if se_parts:
            title_bits.append("".join(se_parts))
        if item.plex.title:
            title_bits.append(item.plex.title)
        if title_bits:
            primary_title = " - ".join(title_bits)
    return primary_title


def _build_point_text(item: AggregatedItem) -> str:
    """Return the vector text for ``item``."""

    parts = [
        _format_primary_title(item),
        item.plex.summary or "",
        item.tmdb.overview if item.tmdb and hasattr(item.tmdb, "overview") else "",
        item.imdb.plot if item.imdb else "",
    ]
    directors_text = ", ".join(p.tag for p in item.plex.directors if p.tag)
    writers_text = ", ".join(p.tag for p in item.plex.writers if p.tag)
    actors_text = ", ".join(p.tag for p in item.plex.actors if p.tag)
    if directors_text:
        parts.append(f"Directed by {directors_text}")
    if writers_text:
        parts.append(f"Written by {writers_text}")
    if actors_text:
        parts.append(f"Starring {actors_text}")
    if item.plex.tagline:
        parts.append(item.plex.tagline)
    if item.tmdb and hasattr(item.tmdb, "tagline"):
        tagline = getattr(item.tmdb, "tagline", None)
        if tagline:
            parts.append(tagline)
    if item.tmdb and hasattr(item.tmdb, "reviews"):
        parts.extend(r.get("content", "") for r in getattr(item.tmdb, "reviews", []))
    return "\n".join(p for p in parts if p)


class _BaseQdrantPayload(TypedDict):
    data: dict[str, JSONValue]
    title: str
    type: str


class QdrantPayload(_BaseQdrantPayload, total=False):
    title_slug: str
    show_title: str
    show_title_slug: str
    season_title: str
    season_number: int
    episode_number: int
    actors: list[str]
    directors: list[str]
    writers: list[str]
    genres: list[str]
    collections: list[str]
    summary: str
    overview: str
    plot: str
    tagline: str
    reviews: list[str]
    year: int
    added_at: int


def _build_point_payload(item: AggregatedItem) -> QdrantPayload:
    """Construct the Qdrant payload for ``item``."""

    payload: QdrantPayload = {
        "data": item.model_dump(mode="json"),
        "title": item.plex.title,
        "type": item.plex.type,
    }
    title_slug = slugify(strip_leading_article(item.plex.title))
    if title_slug:
        payload["title_slug"] = title_slug
    if item.plex.type == "episode":
        if item.plex.show_title:
            payload["show_title"] = item.plex.show_title
            show_slug = slugify(strip_leading_article(item.plex.show_title))
            if show_slug:
                payload["show_title_slug"] = show_slug
        if item.plex.season_title:
            payload["season_title"] = item.plex.season_title
        if item.plex.season_number is not None:
            payload["season_number"] = item.plex.season_number
        if item.plex.episode_number is not None:
            payload["episode_number"] = item.plex.episode_number
    if item.plex.actors:
        payload["actors"] = [p.tag for p in item.plex.actors if p.tag]
    if item.plex.directors:
        payload["directors"] = [p.tag for p in item.plex.directors if p.tag]
    if item.plex.writers:
        payload["writers"] = [p.tag for p in item.plex.writers if p.tag]
    if item.plex.genres:
        payload["genres"] = item.plex.genres
    if item.plex.collections:
        payload["collections"] = item.plex.collections
    summary = item.plex.summary
    if summary:
        payload["summary"] = summary
    overview = getattr(item.tmdb, "overview", None) if item.tmdb else None
    if overview:
        payload["overview"] = overview
    plot = item.imdb.plot if item.imdb else None
    if plot:
        payload["plot"] = plot
    taglines = [item.plex.tagline]
    if item.tmdb and hasattr(item.tmdb, "tagline"):
        taglines.append(getattr(item.tmdb, "tagline", None))
    taglines = [t for t in taglines if t]
    if taglines:
        payload["tagline"] = "\n".join(dict.fromkeys(taglines))
    if item.tmdb and hasattr(item.tmdb, "reviews"):
        review_texts = [r.get("content", "") for r in getattr(item.tmdb, "reviews", [])]
        review_texts = [r for r in review_texts if r]
        if review_texts:
            payload["reviews"] = review_texts
    if item.plex.year is not None:
        payload["year"] = item.plex.year
    if item.plex.added_at is not None:
        added = item.plex.added_at
        if hasattr(added, "timestamp"):
            payload["added_at"] = int(added.timestamp())
    return payload


def _point_id_from_rating_key(rating_key: str) -> int | str:
    """Convert a Plex rating key into the corresponding Qdrant point ID."""

    rating_key = str(rating_key)
    return int(rating_key) if rating_key.isdigit() else rating_key


def _normalise_point_id(point_id: int | str) -> str:
    """Return a stable string representation for comparing point identifiers."""

    return str(point_id)


def build_point(
    item: AggregatedItem,
    dense_model_name: str,
    sparse_model_name: str,
) -> models.PointStruct:
    """Build a Qdrant point for ``item`` using the configured model names."""

    text = _build_point_text(item)
    payload = _build_point_payload(item)
    point_id = _point_id_from_rating_key(item.plex.rating_key)
    return models.PointStruct(
        id=point_id,
        vector={
            "dense": models.Document(text=text, model=dense_model_name),
            "sparse": models.Document(text=text, model=sparse_model_name),
        },
        payload=payload,
    )


_EXISTING_POINT_RETRY_ATTEMPTS = 3
_EXISTING_POINT_RETRY_BACKOFF_S = 0.1


_SINGLE_RETRIEVE_MAX_ATTEMPTS = 3
_SINGLE_RETRIEVE_INITIAL_BACKOFF_S = 0.1


async def _existing_point_ids(
    client: AsyncQdrantClient,
    collection_name: str,
    point_ids: Sequence[int | str],
) -> set[str]:
    """Return the subset of ``point_ids`` already present in Qdrant."""

    if not point_ids:
        return set()

    unique_ids: list[int | str] = list(dict.fromkeys(point_ids))
    total = len(unique_ids)

    async def retrieve_range(start: int, end: int) -> list[models.Record]:
        if start >= end:
            return []

        ids = unique_ids[start:end]
        span = end - start

        if span == 1:
            backoff = _SINGLE_RETRIEVE_INITIAL_BACKOFF_S
            for attempt in range(_SINGLE_RETRIEVE_MAX_ATTEMPTS):
                try:
                    result = await client.retrieve(
                        collection_name=collection_name,
                        ids=ids,
                        with_payload=False,
                    )
                    return list(result or [])
                except Exception:
                    if attempt == _SINGLE_RETRIEVE_MAX_ATTEMPTS - 1:
                        logger.exception(
                            "Failed to check existing Qdrant point %s in collection %s.",
                            _normalise_point_id(unique_ids[start]),
                            collection_name,
                        )
                        return []
                    logger.debug(
                        "Retrying single Qdrant retrieve for %s (attempt %d/%d).",
                        _normalise_point_id(unique_ids[start]),
                        attempt + 1,
                        _SINGLE_RETRIEVE_MAX_ATTEMPTS,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
            return []  # pragma: no cover - loop exits via return

        try:
            result = await client.retrieve(
                collection_name=collection_name,
                ids=ids,
                with_payload=False,
            )
            return list(result or [])
        except Exception:
            midpoint = start + span // 2
            logger.debug(
                "Retrying Qdrant retrieve for %d ids in smaller batches (total=%d).",
                span,
                total,
            )
            left = await retrieve_range(start, midpoint)
            right = await retrieve_range(midpoint, end)
            return left + right

    records = await retrieve_range(0, total)

    existing: set[str] = set()
    for record in records or []:
        record_id = getattr(record, "id", None)
        if record_id is None:
            continue
        existing.add(_normalise_point_id(record_id))
    return existing


def _is_embedding_model_failure(error: Exception) -> bool:
    message = str(error)
    return "Could not load model" in message or "Could not find config.json" in message


def _materialize_points(points: Sequence[models.PointStruct]) -> list[models.PointStruct]:
    materialized: list[models.PointStruct] = []
    for point in points:
        vector = point.vector or {}
        dense = vector.get("dense")
        sparse = vector.get("sparse")
        if isinstance(dense, models.Document):
            size, _ = _resolve_dense_model_params(dense.model)
            dense_vector: list[float] | models.Vector = [0.0] * size
        else:
            dense_vector = dense
        if isinstance(sparse, models.Document):
            sparse_vector: models.SparseVector | None = models.SparseVector(
                indices=[], values=[]
            )
        else:
            sparse_vector = sparse
        materialized.append(
            models.PointStruct(
                id=point.id,
                vector={"dense": dense_vector, "sparse": sparse_vector},
                payload=point.payload,
            )
        )
    return materialized


def _coerce_tmdb_match_value(tmdb_id: str | None) -> int | str | None:
    if tmdb_id is None:
        return None
    tmdb_id_str = str(tmdb_id)
    if tmdb_id_str.isdigit():
        return int(tmdb_id_str)
    return tmdb_id_str


async def _find_record_by_external_ids(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    imdb_id: str | None,
    tmdb_id: str | None,
    plex_guid: str | None,
) -> models.Record | None:
    """Return the first Qdrant record matching the provided external IDs."""

    conditions: list[models.FieldCondition] = []
    if imdb_id:
        conditions.append(
            models.FieldCondition(
                key="data.imdb.id",
                match=models.MatchValue(value=imdb_id),
            )
        )
    tmdb_value = _coerce_tmdb_match_value(tmdb_id)
    if tmdb_value is not None:
        conditions.append(
            models.FieldCondition(
                key="data.tmdb.id",
                match=models.MatchValue(value=tmdb_value),
            )
        )
    if plex_guid:
        conditions.append(
            models.FieldCondition(
                key="data.plex.guid",
                match=models.MatchValue(value=plex_guid),
            )
        )

    if not conditions:
        return None

    records, _ = await client.scroll(
        collection_name=collection_name,
        scroll_filter=models.Filter(must=conditions),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None
    return records[0]


def rehydrate_aggregated_item(payload: Mapping[str, JSONValue]) -> AggregatedItem:
    """Rehydrate an AggregatedItem from a Qdrant payload."""

    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("Qdrant payload missing expected data payload.")
    return AggregatedItem.model_validate(data)


def _chunk(seq: Sequence[models.PointStruct], size: int) -> Iterable[Sequence[models.PointStruct]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def _upsert_batch(
    client: AsyncQdrantClient,
    collection: str,
    batch: Sequence[models.PointStruct],
    *,
    max_retries: int,
    initial_backoff_s: float,
    retry_queue: Optional[asyncio.Queue[List[models.PointStruct]]],
    batch_idx: int,
    total_batches: int,
) -> int:
    """Returns number of points successfully upserted (len(batch) or 0 if handed to retry_queue)."""
    attempt = 0
    batch_list = list(batch)
    while True:
        try:
            # Fast path: weak ordering, don't block the write pipeline
            await client.upsert(
                collection_name=collection,
                points=batch_list,
                wait=False,
                ordering=WriteOrdering.WEAK,
            )
            return len(batch)
        except Exception as e:
            if _is_embedding_model_failure(e):
                inner = getattr(client, "_client", None)
                if inner is not None and hasattr(inner, "upsert"):
                    try:
                        await inner.upsert(
                            collection_name=collection,
                            points=_materialize_points(batch_list),
                        )
                    except Exception:
                        logger.exception(
                            "Fallback upsert failed for Qdrant batch %d/%d after embedding model error.",
                            batch_idx,
                            total_batches,
                        )
                        return 0
                    logger.warning(
                        "Upserted Qdrant batch %d/%d with placeholder vectors due to missing embedding model.",
                        batch_idx,
                        total_batches,
                    )
                    return len(batch_list)
                logger.error(
                    "Skipping Qdrant batch %d/%d due to missing embedding model: %s",
                    batch_idx,
                    total_batches,
                    e,
                )
                return 0
            attempt += 1
            if attempt > max_retries:
                # Shove it to the retry queue and move on
                if retry_queue is not None:
                    try:
                        await retry_queue.put(list(batch_list))
                        logger.error(
                            "Batch %d/%d permanently failed after %d attempts; queued for retry: %s",
                            batch_idx,
                            total_batches,
                            attempt - 1,
                            e,
                        )
                    except Exception:
                        logger.exception("Failed to enqueue failed batch %d/%d", batch_idx, total_batches)
                else:
                    logger.exception(
                        "Batch %d/%d permanently failed after %d attempts (no retry queue): %s",
                        batch_idx,
                        total_batches,
                        attempt - 1,
                        e,
                    )
                return 0

            # Exponential backoff with jitter
            sleep_s = (initial_backoff_s * (2 ** (attempt - 1))) * (0.5 + random.random())
            # Cap backoff so we don't take a nap long enough to miss a birthday
            sleep_s = min(sleep_s, 10.0)
            logger.warning(
                "Upsert batch %d/%d failed (attempt %d/%d): %s; backing off %.2fs",
                batch_idx,
                total_batches,
                attempt,
                max_retries,
                e,
                sleep_s,
            )
            await asyncio.sleep(sleep_s)


async def _bounded_gather(coros, *, limit: int):
    sem = asyncio.Semaphore(limit)

    async def _runner(coro):
        async with sem:
            return await coro

    return await asyncio.gather(*(_runner(c) for c in coros), return_exceptions=False)


async def _upsert_in_batches(
    client: AsyncQdrantClient,
    collection_name: str,
    points: Sequence[models.PointStruct],
    *,
    batch_size: int = 2000,
    concurrency: int = 8,
    max_retries: int = 4,
    initial_backoff_s: float = 0.25,
    retry_queue: Optional[asyncio.Queue[List[models.PointStruct]]] = None,
) -> None:
    """
    High-throughput upsert:
      - batches points
      - runs upserts concurrently with weak ordering and wait=False
      - retries transient failures with exponential backoff + jitter
      - optionally pushes perma-failed batches to retry_queue
    """

    total = len(points)
    if total == 0:
        logger.info("No points to upsert. Graceful idleness. Touch grass.")
        return

    batches = list(_chunk(points, batch_size))
    total_batches = len(batches)
    t0 = time.perf_counter()

    # Submit tasks
    tasks = [
        _upsert_batch(
            client,
            collection_name,
            batch,
            max_retries=max_retries,
            initial_backoff_s=initial_backoff_s,
            retry_queue=retry_queue,
            batch_idx=i + 1,
            total_batches=total_batches,
        )
        for i, batch in enumerate(batches)
    ]

    # Concurrency with backpressure
    completed_points = 0
    for i in range(0, len(tasks), concurrency):
        chunk = tasks[i : i + concurrency]
        # Run a window of tasks
        results = await _bounded_gather(chunk, limit=concurrency)
        completed_points += sum(results)

        done = min(i + concurrency, len(tasks))
        pct = (done / total_batches) * 100
        elapsed = time.perf_counter() - t0
        rps = (completed_points / elapsed) if elapsed > 0 else math.inf
        logger.info(
            "Upsert progress: %d/%d batches (%.1f%%), %d/%d points, ~%.0f pts/s",
            done,
            total_batches,
            pct,
            completed_points,
            total,
            rps,
        )

    elapsed = time.perf_counter() - t0
    rps = (completed_points / elapsed) if elapsed > 0 else math.inf
    logger.info(
        "Upsert complete: %d/%d points in %.2fs (%.0f pts/s).",
        completed_points,
        total,
        elapsed,
        rps,
    )


async def _process_qdrant_retry_queue(
    client: AsyncQdrantClient,
    collection_name: str,
    retry_queue: asyncio.Queue[list[models.PointStruct]],
    *,
    config: "QdrantRuntimeConfig",
) -> tuple[int, int]:
    """Retry failed Qdrant batches with exponential backoff.

    Returns a tuple containing the number of points that were retried successfully
    and the number that still failed after exhausting ``config.retry_attempts``.
    """

    if retry_queue.empty():
        return 0, 0

    pending = retry_queue.qsize()
    logger.info("Retrying %d failed Qdrant batches", pending)
    succeeded_points = 0
    failed_points = 0
    while not retry_queue.empty():
        batch = await retry_queue.get()
        batch_size = len(batch)
        for attempt in range(1, config.retry_attempts + 1):
            try:
                await client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
            except Exception:
                logger.exception(
                    "Retry %d/%d failed for Qdrant batch of %d points",
                    attempt,
                    config.retry_attempts,
                    batch_size,
                )
                if attempt == config.retry_attempts:
                    logger.error(
                        "Giving up on Qdrant batch after %d attempts; %d points were not indexed",
                        config.retry_attempts,
                        batch_size,
                    )
                    failed_points += batch_size
                    break

                next_attempt = attempt + 1
                await asyncio.sleep(config.retry_backoff * next_attempt)
            else:
                logger.info(
                    "Successfully retried Qdrant batch of %d points on attempt %d",
                    batch_size,
                    attempt,
                )
                succeeded_points += batch_size
                break

    return succeeded_points, failed_points

__all__ = [
    "_DENSE_MODEL_PARAMS",
    "_resolve_dense_model_params",
    "_is_local_qdrant",
    "_ensure_collection",
    "_build_point_text",
    "_build_point_payload",
    "_point_id_from_rating_key",
    "_normalise_point_id",
    "_existing_point_ids",
    "_find_record_by_external_ids",
    "rehydrate_aggregated_item",
    "QdrantPayload",
    "build_point",
    "_upsert_in_batches",
    "_process_qdrant_retry_queue",
]
