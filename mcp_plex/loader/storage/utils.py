from __future__ import annotations

import asyncio
import logging
import warnings
from typing import Sequence

from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from mcp_plex.common.types import AggregatedItem

from ..utils import is_local_qdrant


def format_primary_title(item: AggregatedItem) -> str:
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


def build_point_text(item: AggregatedItem) -> str:
    """Return the vector text for ``item``."""

    parts = [
        format_primary_title(item),
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


def build_point_payload(item: AggregatedItem) -> dict[str, object]:
    """Construct the Qdrant payload for ``item``."""

    payload: dict[str, object] = {
        "data": item.model_dump(mode="json"),
        "title": item.plex.title,
        "type": item.plex.type,
    }
    if item.plex.type == "episode":
        if item.plex.show_title:
            payload["show_title"] = item.plex.show_title
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


def build_point(
    item: AggregatedItem,
    dense_model_name: str,
    sparse_model_name: str,
) -> models.PointStruct:
    """Build a Qdrant point for ``item`` using the configured model names."""

    text = build_point_text(item)
    payload = build_point_payload(item)
    point_id: int | str = (
        int(item.plex.rating_key)
        if item.plex.rating_key.isdigit()
        else item.plex.rating_key
    )
    return models.PointStruct(
        id=point_id,
        vector={
            "dense": models.Document(text=text, model=dense_model_name),
            "sparse": models.Document(text=text, model=sparse_model_name),
        },
        payload=payload,
    )


async def upsert_in_batches(
    client: AsyncQdrantClient,
    collection_name: str,
    points: Sequence[models.PointStruct],
    *,
    batch_size: int,
    retry_queue: asyncio.Queue[list[models.PointStruct]] | None = None,
    logger: logging.Logger,
) -> None:
    """Upsert points into Qdrant in batches, logging HTTP errors."""

    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i : i + batch_size]
        try:
            await client.upsert(collection_name=collection_name, points=batch)
        except Exception:
            logger.exception(
                "Failed to upsert batch %d-%d", i, i + len(batch)
            )
            if retry_queue is not None:
                await retry_queue.put(list(batch))
        else:
            logger.info(
                "Upserted %d/%d points", min(i + len(batch), total), total
            )


async def process_qdrant_retry_queue(
    client: AsyncQdrantClient,
    collection_name: str,
    retry_queue: asyncio.Queue[list[models.PointStruct]],
    logger: logging.Logger,
    *,
    max_attempts: int,
    backoff: float,
) -> None:
    """Retry failed Qdrant batches with exponential backoff."""

    if retry_queue.empty():
        return

    pending = retry_queue.qsize()
    logger.info("Retrying %d failed Qdrant batches", pending)
    while not retry_queue.empty():
        batch = await retry_queue.get()
        attempt = 1
        while attempt <= max_attempts:
            try:
                await client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
            except Exception:
                logger.exception(
                    "Retry %d/%d failed for Qdrant batch of %d points",
                    attempt,
                    max_attempts,
                    len(batch),
                )
                attempt += 1
                if attempt > max_attempts:
                    logger.error(
                        "Giving up on Qdrant batch after %d attempts; %d points were not indexed",
                        max_attempts,
                        len(batch),
                    )
                    break
                await asyncio.sleep(backoff * attempt)
                continue
            else:
                logger.info(
                    "Successfully retried Qdrant batch of %d points on attempt %d",
                    len(batch),
                    attempt,
                )
                break


async def ensure_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    dense_size: int,
    dense_distance: models.Distance,
    logger: logging.Logger,
) -> None:
    """Create the collection and payload indexes if they do not already exist."""

    created_collection = False
    if not await client.collection_exists(collection_name):
        await client.create_collection(
            collection_name=collection_name,
            vectors_config={"dense": models.VectorParams(size=dense_size, distance=dense_distance)},
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        created_collection = True

    if not created_collection:
        return

    suppress_payload_warning = is_local_qdrant(client)

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
    for field, schema in (
        ("title", text_index),
        ("type", models.PayloadSchemaType.KEYWORD),
        ("year", models.PayloadSchemaType.INTEGER),
        ("added_at", models.PayloadSchemaType.INTEGER),
        ("actors", models.PayloadSchemaType.KEYWORD),
        ("directors", models.PayloadSchemaType.KEYWORD),
        ("writers", models.PayloadSchemaType.KEYWORD),
        ("genres", models.PayloadSchemaType.KEYWORD),
        ("show_title", models.PayloadSchemaType.KEYWORD),
        ("season_number", models.PayloadSchemaType.INTEGER),
        ("episode_number", models.PayloadSchemaType.INTEGER),
        ("collections", models.PayloadSchemaType.KEYWORD),
        ("summary", text_index),
        ("overview", text_index),
        ("plot", text_index),
        ("tagline", text_index),
        ("reviews", text_index),
        ("data.plex.rating_key", models.PayloadSchemaType.KEYWORD),
        ("data.imdb.id", models.PayloadSchemaType.KEYWORD),
        ("data.tmdb.id", models.PayloadSchemaType.INTEGER),
    ):
        await _create_index(field, schema)

    logger.info("Ensured collection %s exists", collection_name)


__all__ = [
    "build_point",
    "build_point_payload",
    "build_point_text",
    "ensure_collection",
    "format_primary_title",
    "process_qdrant_retry_queue",
    "upsert_in_batches",
]
