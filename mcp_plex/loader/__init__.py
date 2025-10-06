"""Loader orchestration utilities and staged pipeline helpers."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, List, Optional, Sequence, TypeVar

import click
import httpx
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from .imdb_cache import IMDbCache
from .pipeline.channels import (
    IMDbRetryQueue,
    INGEST_DONE,
    IngestBatch,
    IngestQueue,
    MovieBatch,
    EpisodeBatch,
    PersistenceQueue,
    SampleBatch,
    chunk_sequence,
    require_positive,
)
from .pipeline.orchestrator import LoaderOrchestrator
from .pipeline.persistence import PersistenceStage as _PersistenceStage
from ..common.types import (
    AggregatedItem,
    IMDbTitle,
    PlexGuid,
    PlexItem,
    PlexPerson,
    TMDBMovie,
    TMDBShow,
)

try:  # Only import plexapi when available; the sample data mode does not require it.
    from plexapi.base import PlexPartialObject
    from plexapi.server import PlexServer
except Exception:
    PlexServer = None  # type: ignore[assignment]
    PlexPartialObject = object  # type: ignore[assignment]


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*'mcp_plex\\.loader' found in sys.modules after import of package 'mcp_plex'.*",
    category=RuntimeWarning,
)

T = TypeVar("T")

_imdb_cache: IMDbCache | None = None
_imdb_max_retries: int = 3
_imdb_backoff: float = 1.0
_imdb_retry_queue: "_IMDbRetryQueue" | None = None
_imdb_batch_limit: int = 5
_imdb_requests_per_window: int | None = None
_imdb_window_seconds: float = 1.0
_imdb_throttle: Any = None
_qdrant_batch_size: int = 1000
_qdrant_upsert_buffer_size: int = 200
_qdrant_max_concurrent_upserts: int = 4
_qdrant_retry_attempts: int = 3
_qdrant_retry_backoff: float = 1.0

# Backwards-compatible aliases while callers migrate to shared pipeline exports.
_MovieBatch = MovieBatch
_EpisodeBatch = EpisodeBatch
_SampleBatch = SampleBatch
_IngestBatch = IngestBatch
_IngestQueue = IngestQueue
_PersistenceQueue = PersistenceQueue
_require_positive = require_positive
_chunk_sequence = chunk_sequence
_IMDbRetryQueue = IMDbRetryQueue
_INGEST_DONE = INGEST_DONE


def _is_local_qdrant(client: AsyncQdrantClient) -> bool:
    """Return ``True`` if *client* targets an in-process Qdrant instance."""

    inner = getattr(client, "_client", None)
    return bool(inner) and inner.__class__.__module__.startswith(
        "qdrant_client.local"
    )


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


def _get_imdb_throttle() -> Any:
    """Return the shared IMDb rate limiter instance if configured."""

    global _imdb_throttle
    if _imdb_requests_per_window is None:
        return None
    from .pipeline import enrichment as enrichment_mod

    if _imdb_throttle is None:
        _imdb_throttle = enrichment_mod._RequestThrottler(
            limit=_imdb_requests_per_window,
            interval=float(_imdb_window_seconds),
        )
    return _imdb_throttle


async def _fetch_imdb(client: httpx.AsyncClient, imdb_id: str) -> Optional[IMDbTitle]:
    """Fetch metadata for an IMDb ID with caching, retry, and throttling."""

    from .pipeline import enrichment as enrichment_mod

    return await enrichment_mod._fetch_imdb(
        client,
        imdb_id,
        cache=_imdb_cache,
        throttle=_get_imdb_throttle(),
        max_retries=_imdb_max_retries,
        backoff=_imdb_backoff,
        retry_queue=_imdb_retry_queue,
    )


def _load_imdb_retry_queue(path: Path) -> None:
    """Populate the retry queue from a JSON file if it exists."""

    global _imdb_retry_queue
    ids: list[str] = []
    if path.exists():
        try:
            data = json.loads(path.read_text())
            if isinstance(data, list):
                ids = [str(imdb_id) for imdb_id in data]
            else:
                logger.warning(
                    "IMDb retry queue file %s did not contain a list; ignoring its contents",
                    path,
                )
        except Exception:
            logger.exception("Failed to load IMDb retry queue from %s", path)
    _imdb_retry_queue = _IMDbRetryQueue(ids)


async def _process_imdb_retry_queue(client: httpx.AsyncClient) -> None:
    """Attempt to fetch queued IMDb IDs, re-queueing failures."""

    if _imdb_retry_queue is None or _imdb_retry_queue.empty():
        return
    size = _imdb_retry_queue.qsize()
    for _ in range(size):
        imdb_id = await _imdb_retry_queue.get()
        title = await _fetch_imdb(client, imdb_id)
        if title is None:
            await _imdb_retry_queue.put(imdb_id)


def _persist_imdb_retry_queue(path: Path) -> None:
    """Persist the retry queue to disk."""

    if _imdb_retry_queue is None:
        return
    path.write_text(json.dumps(_imdb_retry_queue.snapshot()))


async def _upsert_in_batches(
    client: AsyncQdrantClient,
    collection_name: str,
    points: Sequence[models.PointStruct],
    *,
    retry_queue: asyncio.Queue[list[models.PointStruct]] | None = None,
) -> None:
    """Upsert points into Qdrant in batches, logging HTTP errors."""

    total = len(points)
    for i in range(0, total, _qdrant_batch_size):
        batch = points[i : i + _qdrant_batch_size]
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


async def _process_qdrant_retry_queue(
    client: AsyncQdrantClient,
    collection_name: str,
    retry_queue: asyncio.Queue[list[models.PointStruct]],
) -> None:
    """Retry failed Qdrant batches with exponential backoff."""

    if retry_queue.empty():
        return

    pending = retry_queue.qsize()
    logger.info("Retrying %d failed Qdrant batches", pending)
    while not retry_queue.empty():
        batch = await retry_queue.get()
        attempt = 1
        while attempt <= _qdrant_retry_attempts:
            try:
                await client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )
            except Exception:
                logger.exception(
                    "Retry %d/%d failed for Qdrant batch of %d points",
                    attempt,
                    _qdrant_retry_attempts,
                    len(batch),
                )
                attempt += 1
                if attempt > _qdrant_retry_attempts:
                    logger.error(
                        "Giving up on Qdrant batch after %d attempts; %d points were not indexed",
                        _qdrant_retry_attempts,
                        len(batch),
                    )
                    break
                await asyncio.sleep(_qdrant_retry_backoff * attempt)
                continue
            else:
                logger.info(
                    "Successfully retried Qdrant batch of %d points on attempt %d",
                    len(batch),
                    attempt,
                )
                break


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
            vectors_config={"dense": models.VectorParams(size=dense_size, distance=dense_distance)},
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
    await _create_index("type", models.PayloadSchemaType.KEYWORD)
    await _create_index("year", models.PayloadSchemaType.INTEGER)
    await _create_index("added_at", models.PayloadSchemaType.INTEGER)
    await _create_index("actors", models.PayloadSchemaType.KEYWORD)
    await _create_index("directors", models.PayloadSchemaType.KEYWORD)
    await _create_index("writers", models.PayloadSchemaType.KEYWORD)
    await _create_index("genres", models.PayloadSchemaType.KEYWORD)
    await _create_index("show_title", models.PayloadSchemaType.KEYWORD)
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


def _build_point_payload(item: AggregatedItem) -> dict[str, object]:
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

    text = _build_point_text(item)
    payload = _build_point_payload(item)
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


def _load_from_sample(sample_dir: Path) -> List[AggregatedItem]:
    """Load items from local sample JSON files."""

    results: List[AggregatedItem] = []
    movie_dir = sample_dir / "movie"
    episode_dir = sample_dir / "episode"

    # Movie sample
    with (movie_dir / "plex.json").open("r", encoding="utf-8") as f:
        movie_data = json.load(f)["MediaContainer"]["Metadata"][0]
    plex_movie = PlexItem(
        rating_key=str(movie_data.get("ratingKey", "")),
        guid=str(movie_data.get("guid", "")),
        type=movie_data.get("type", "movie"),
        title=movie_data.get("title", ""),
        summary=movie_data.get("summary"),
        year=movie_data.get("year"),
        added_at=movie_data.get("addedAt"),
        guids=[PlexGuid(id=g["id"]) for g in movie_data.get("Guid", [])],
        thumb=movie_data.get("thumb"),
        art=movie_data.get("art"),
        tagline=movie_data.get("tagline"),
        content_rating=movie_data.get("contentRating"),
        directors=[
            PlexPerson(id=d.get("id", 0), tag=d.get("tag", ""), thumb=d.get("thumb"))
            for d in movie_data.get("Director", [])
        ],
        writers=[
            PlexPerson(id=w.get("id", 0), tag=w.get("tag", ""), thumb=w.get("thumb"))
            for w in movie_data.get("Writer", [])
        ],
        actors=[
            PlexPerson(
                id=a.get("id", 0),
                tag=a.get("tag", ""),
                role=a.get("role"),
                thumb=a.get("thumb"),
            )
            for a in movie_data.get("Role", [])
        ],
        genres=[g.get("tag", "") for g in movie_data.get("Genre", []) if g.get("tag")],
        collections=[
            c.get("tag", "")
            for key in ("Collection", "Collections")
            for c in movie_data.get(key, []) or []
            if c.get("tag")
        ],
    )
    with (movie_dir / "imdb.json").open("r", encoding="utf-8") as f:
        imdb_movie = IMDbTitle.model_validate(json.load(f))
    with (movie_dir / "tmdb.json").open("r", encoding="utf-8") as f:
        tmdb_movie = TMDBMovie.model_validate(json.load(f))
    results.append(AggregatedItem(plex=plex_movie, imdb=imdb_movie, tmdb=tmdb_movie))

    # Episode sample
    with (episode_dir / "plex.tv.json").open("r", encoding="utf-8") as f:
        episode_data = json.load(f)["MediaContainer"]["Metadata"][0]
    plex_episode = PlexItem(
        rating_key=str(episode_data.get("ratingKey", "")),
        guid=str(episode_data.get("guid", "")),
        type=episode_data.get("type", "episode"),
        title=episode_data.get("title", ""),
        show_title=episode_data.get("grandparentTitle"),
        season_title=episode_data.get("parentTitle"),
        season_number=episode_data.get("parentIndex"),
        episode_number=episode_data.get("index"),
        summary=episode_data.get("summary"),
        year=episode_data.get("year"),
        added_at=episode_data.get("addedAt"),
        guids=[PlexGuid(id=g["id"]) for g in episode_data.get("Guid", [])],
        thumb=episode_data.get("thumb"),
        art=episode_data.get("art"),
        tagline=episode_data.get("tagline"),
        content_rating=episode_data.get("contentRating"),
        directors=[
            PlexPerson(id=d.get("id", 0), tag=d.get("tag", ""), thumb=d.get("thumb"))
            for d in episode_data.get("Director", [])
        ],
        writers=[
            PlexPerson(id=w.get("id", 0), tag=w.get("tag", ""), thumb=w.get("thumb"))
            for w in episode_data.get("Writer", [])
        ],
        actors=[
            PlexPerson(
                id=a.get("id", 0),
                tag=a.get("tag", ""),
                role=a.get("role"),
                thumb=a.get("thumb"),
            )
            for a in episode_data.get("Role", [])
        ],
        genres=[g.get("tag", "") for g in episode_data.get("Genre", []) if g.get("tag")],
        collections=[
            c.get("tag", "")
            for key in ("Collection", "Collections")
            for c in episode_data.get(key, []) or []
            if c.get("tag")
        ],
    )
    with (episode_dir / "imdb.tv.json").open("r", encoding="utf-8") as f:
        imdb_episode = IMDbTitle.model_validate(json.load(f))
    with (episode_dir / "tmdb.tv.json").open("r", encoding="utf-8") as f:
        tmdb_show = TMDBShow.model_validate(json.load(f))
    results.append(AggregatedItem(plex=plex_episode, imdb=imdb_episode, tmdb=tmdb_show))

    return results


def _build_loader_orchestrator(
    *,
    client: AsyncQdrantClient,
    collection_name: str,
    dense_model_name: str,
    sparse_model_name: str,
    tmdb_api_key: str | None,
    sample_items: Sequence[AggregatedItem] | None,
    plex_server: PlexServer | None,
    plex_chunk_size: int,
    enrichment_batch_size: int,
    enrichment_workers: int,
    upsert_buffer_size: int,
    max_concurrent_upserts: int,
    imdb_retry_queue: IMDbRetryQueue | None = None,
) -> tuple[LoaderOrchestrator, list[AggregatedItem], asyncio.Queue[list[models.PointStruct]]]:
    """Wire the staged loader pipeline and return the orchestrator helpers."""

    from .pipeline.ingestion import IngestionStage
    from .pipeline.enrichment import EnrichmentStage

    ingest_queue: IngestQueue = IngestQueue(maxsize=enrichment_workers * 2)
    persistence_queue: PersistenceQueue = PersistenceQueue()

    imdb_queue = imdb_retry_queue
    if imdb_queue is None:
        global _imdb_retry_queue
        if _imdb_retry_queue is None:
            _imdb_retry_queue = _IMDbRetryQueue()
        imdb_queue = _imdb_retry_queue

    upsert_capacity = asyncio.Semaphore(max_concurrent_upserts)
    qdrant_retry_queue: asyncio.Queue[list[models.PointStruct]] = asyncio.Queue()
    items: list[AggregatedItem] = []
    upserted = 0
    upsert_start = time.perf_counter()

    async def _upsert_aggregated(
        batch: Sequence[AggregatedItem],
    ) -> None:
        if not batch:
            return
        items.extend(batch)
        points = [
            build_point(item, dense_model_name, sparse_model_name)
            for item in batch
        ]
        for point_chunk in chunk_sequence(points, upsert_buffer_size):
            await _upsert_in_batches(
                client,
                collection_name,
                list(point_chunk),
                retry_queue=qdrant_retry_queue,
            )

    def _record_upsert(worker_id: int, batch_size: int, queue_size: int) -> None:
        nonlocal upserted, upsert_start
        if upserted == 0:
            upsert_start = time.perf_counter()
        upserted += batch_size
        elapsed = time.perf_counter() - upsert_start
        rate = upserted / elapsed if elapsed > 0 else 0.0
        logger.info(
            "Upsert worker %d processed %d items (%.2f items/sec, queue size=%d)",
            worker_id,
            upserted,
            rate,
            queue_size,
        )

    ingestion_stage = IngestionStage(
        plex_server=plex_server,
        sample_items=sample_items,
        movie_batch_size=plex_chunk_size,
        episode_batch_size=plex_chunk_size,
        sample_batch_size=enrichment_batch_size,
        output_queue=ingest_queue,
        completion_sentinel=INGEST_DONE,
    )

    enrichment_stage = EnrichmentStage(
        http_client_factory=lambda: httpx.AsyncClient(timeout=30),
        tmdb_api_key=tmdb_api_key or "",
        ingest_queue=ingest_queue,
        persistence_queue=persistence_queue,
        imdb_retry_queue=imdb_queue,
        movie_batch_size=enrichment_batch_size,
        episode_batch_size=enrichment_batch_size,
        imdb_cache=_imdb_cache,
        imdb_max_retries=_imdb_max_retries,
        imdb_backoff=_imdb_backoff,
        imdb_batch_limit=_imdb_batch_limit,
        imdb_requests_per_window=_imdb_requests_per_window,
        imdb_window_seconds=_imdb_window_seconds,
    )

    persistence_stage = _PersistenceStage(
        client=client,
        collection_name=collection_name,
        dense_vector_name=dense_model_name,
        sparse_vector_name=sparse_model_name,
        persistence_queue=persistence_queue,
        retry_queue=qdrant_retry_queue,
        upsert_semaphore=upsert_capacity,
        upsert_buffer_size=upsert_buffer_size,
        upsert_fn=_upsert_aggregated,
        on_batch_complete=_record_upsert,
        worker_count=max_concurrent_upserts,
    )

    orchestrator = LoaderOrchestrator(
        ingestion_stage=ingestion_stage,
        enrichment_stage=enrichment_stage,
        persistence_stage=persistence_stage,
        ingest_queue=ingest_queue,
        persistence_queue=persistence_queue,
        persistence_worker_count=max_concurrent_upserts,
    )

    return orchestrator, items, persistence_stage.retry_queue


async def run(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    qdrant_host: Optional[str] = None,
    qdrant_port: int = 6333,
    qdrant_grpc_port: int = 6334,
    qdrant_https: bool = False,
    qdrant_prefer_grpc: bool = False,
    dense_model_name: str = "BAAI/bge-small-en-v1.5",
    sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions",
    imdb_cache_path: Path | None = None,
    imdb_max_retries: int = 3,
    imdb_backoff: float = 1.0,
    imdb_queue_path: Path | None = None,
    imdb_requests_per_window: int | None = None,
    imdb_window_seconds: float = 1.0,
    upsert_buffer_size: int = _qdrant_upsert_buffer_size,
    plex_chunk_size: int = 200,
    enrichment_batch_size: int = 100,
    enrichment_workers: int = 4,
) -> None:
    """Core execution logic for the CLI."""

    global _imdb_cache
    global _imdb_max_retries
    global _imdb_backoff
    global _imdb_retry_queue
    global _imdb_requests_per_window
    global _imdb_window_seconds
    global _imdb_throttle
    _imdb_cache = IMDbCache(imdb_cache_path) if imdb_cache_path else None
    _imdb_max_retries = imdb_max_retries
    _imdb_backoff = imdb_backoff
    if imdb_requests_per_window is not None:
        _require_positive(imdb_requests_per_window, name="imdb_requests_per_window")
        if imdb_window_seconds <= 0:
            raise ValueError("imdb_window_seconds must be positive")
    _imdb_requests_per_window = imdb_requests_per_window
    _imdb_window_seconds = imdb_window_seconds
    _imdb_throttle = None
    if imdb_queue_path:
        _load_imdb_retry_queue(imdb_queue_path)
        async with httpx.AsyncClient(timeout=30) as client:
            await _process_imdb_retry_queue(client)
    else:
        _imdb_retry_queue = _IMDbRetryQueue()

    _require_positive(upsert_buffer_size, name="upsert_buffer_size")
    _require_positive(plex_chunk_size, name="plex_chunk_size")
    _require_positive(enrichment_batch_size, name="enrichment_batch_size")
    _require_positive(enrichment_workers, name="enrichment_workers")

    dense_size, dense_distance = _resolve_dense_model_params(dense_model_name)
    if qdrant_url is None and qdrant_host is None:
        qdrant_url = ":memory:"
    client = AsyncQdrantClient(
        location=qdrant_url,
        api_key=qdrant_api_key,
        host=qdrant_host,
        port=qdrant_port,
        grpc_port=qdrant_grpc_port,
        https=qdrant_https,
        prefer_grpc=qdrant_prefer_grpc,
    )
    collection_name = "media-items"
    await _ensure_collection(
        client,
        collection_name,
        dense_size=dense_size,
        dense_distance=dense_distance,
    )

    items: List[AggregatedItem]
    if sample_dir is not None:
        logger.info("Loading sample data from %s", sample_dir)
        sample_items = _load_from_sample(sample_dir)
        orchestrator, items, qdrant_retry_queue = _build_loader_orchestrator(
            client=client,
            collection_name=collection_name,
            dense_model_name=dense_model_name,
            sparse_model_name=sparse_model_name,
            tmdb_api_key=None,
            sample_items=sample_items,
            plex_server=None,
            plex_chunk_size=plex_chunk_size,
            enrichment_batch_size=enrichment_batch_size,
            enrichment_workers=enrichment_workers,
            upsert_buffer_size=upsert_buffer_size,
            max_concurrent_upserts=_qdrant_max_concurrent_upserts,
            imdb_retry_queue=_IMDbRetryQueue(),
        )
        logger.info("Starting staged loader (sample mode)")
        await orchestrator.run()
    else:
        if PlexServer is None:
            raise RuntimeError("plexapi is required for live loading")
        if not plex_url or not plex_token:
            raise RuntimeError("PLEX_URL and PLEX_TOKEN must be provided")
        if not tmdb_api_key:
            raise RuntimeError("TMDB_API_KEY must be provided")
        logger.info("Loading data from Plex server %s", plex_url)
        server = PlexServer(plex_url, plex_token)
        orchestrator, items, qdrant_retry_queue = _build_loader_orchestrator(
            client=client,
            collection_name=collection_name,
            dense_model_name=dense_model_name,
            sparse_model_name=sparse_model_name,
            tmdb_api_key=tmdb_api_key,
            sample_items=None,
            plex_server=server,
            plex_chunk_size=plex_chunk_size,
            enrichment_batch_size=enrichment_batch_size,
            enrichment_workers=enrichment_workers,
            upsert_buffer_size=upsert_buffer_size,
            max_concurrent_upserts=_qdrant_max_concurrent_upserts,
        )
        logger.info("Starting staged loader (Plex mode)")
        await orchestrator.run()
    logger.info("Loaded %d items", len(items))
    if not items:
        logger.info("No points to upsert")

    await _process_qdrant_retry_queue(
        client, collection_name, qdrant_retry_queue
    )

    if imdb_queue_path:
        _persist_imdb_retry_queue(imdb_queue_path)

    json.dump([item.model_dump(mode="json") for item in items], fp=sys.stdout, indent=2)
    sys.stdout.write("\n")


@click.command()
@click.option(
    "--plex-url",
    envvar="PLEX_URL",
    show_envvar=True,
    required=True,
    help="Plex base URL",
)
@click.option(
    "--plex-token",
    envvar="PLEX_TOKEN",
    show_envvar=True,
    required=True,
    help="Plex API token",
)
@click.option(
    "--tmdb-api-key",
    envvar="TMDB_API_KEY",
    show_envvar=True,
    required=True,
    help="TMDb API key",
)
@click.option(
    "--sample-dir",
    type=click.Path(path_type=Path),
    required=False,
    help="Directory containing sample data instead of live Plex access",
)
@click.option(
    "--qdrant-url",
    envvar="QDRANT_URL",
    show_envvar=True,
    required=False,
    help="Qdrant URL or path",
)
@click.option(
    "--qdrant-api-key",
    envvar="QDRANT_API_KEY",
    show_envvar=True,
    required=False,
    help="Qdrant API key",
)
@click.option(
    "--qdrant-host",
    envvar="QDRANT_HOST",
    show_envvar=True,
    required=False,
    help="Qdrant host",
)
@click.option(
    "--qdrant-port",
    envvar="QDRANT_PORT",
    show_envvar=True,
    type=int,
    default=6333,
    show_default=True,
    required=False,
    help="Qdrant HTTP port",
)
@click.option(
    "--qdrant-grpc-port",
    envvar="QDRANT_GRPC_PORT",
    show_envvar=True,
    type=int,
    default=6334,
    show_default=True,
    required=False,
    help="Qdrant gRPC port",
)
@click.option(
    "--qdrant-https/--no-qdrant-https",
    envvar="QDRANT_HTTPS",
    show_envvar=True,
    default=False,
    help="Use HTTPS when connecting to Qdrant",
)
@click.option(
    "--qdrant-prefer-grpc/--no-qdrant-prefer-grpc",
    envvar="QDRANT_PREFER_GRPC",
    show_envvar=True,
    default=False,
    help="Prefer gRPC when connecting to Qdrant",
)
@click.option(
    "--upsert-buffer-size",
    envvar="QDRANT_UPSERT_BUFFER_SIZE",
    show_envvar=True,
    type=int,
    default=_qdrant_upsert_buffer_size,
    show_default=True,
    help="Number of media items to buffer before scheduling an async upsert",
)
@click.option(
    "--plex-chunk-size",
    envvar="PLEX_CHUNK_SIZE",
    show_envvar=True,
    type=int,
    default=200,
    show_default=True,
    help="Number of Plex rating keys to request per fetchItems batch",
)
@click.option(
    "--enrichment-batch-size",
    envvar="ENRICHMENT_BATCH_SIZE",
    show_envvar=True,
    type=int,
    default=100,
    show_default=True,
    help="Number of media items to enrich per metadata batch",
)
@click.option(
    "--enrichment-workers",
    envvar="ENRICHMENT_WORKERS",
    show_envvar=True,
    type=int,
    default=4,
    show_default=True,
    help="Number of concurrent metadata enrichment workers",
)
@click.option(
    "--dense-model",
    envvar="DENSE_MODEL",
    show_envvar=True,
    default="BAAI/bge-small-en-v1.5",
    show_default=True,
    help="Dense embedding model name",
)
@click.option(
    "--sparse-model",
    envvar="SPARSE_MODEL",
    show_envvar=True,
    default="Qdrant/bm42-all-minilm-l6-v2-attentions",
    show_default=True,
    help="Sparse embedding model name",
)
@click.option(
    "--continuous",
    is_flag=True,
    help="Continuously run the loader",
    show_default=True,
    default=False,
    required=False,
)
@click.option(
    "--delay",
    type=float,
    default=300.0,
    show_default=True,
    required=False,
    help="Delay between runs in seconds when using --continuous",
)
@click.option(
    "--imdb-cache",
    envvar="IMDB_CACHE",
    show_envvar=True,
    type=click.Path(path_type=Path),
    default=Path("imdb_cache.json"),
    show_default=True,
    help="Path to persistent IMDb response cache",
)
@click.option(
    "--imdb-max-retries",
    envvar="IMDB_MAX_RETRIES",
    show_envvar=True,
    type=int,
    default=3,
    show_default=True,
    help="Maximum retries for IMDb requests returning HTTP 429",
)
@click.option(
    "--imdb-backoff",
    envvar="IMDB_BACKOFF",
    show_envvar=True,
    type=float,
    default=1.0,
    show_default=True,
    help="Initial backoff delay in seconds for IMDb retries",
)
@click.option(
    "--imdb-requests-per-window",
    envvar="IMDB_REQUESTS_PER_WINDOW",
    show_envvar=True,
    type=int,
    default=None,
    help="Maximum IMDb requests per rate-limit window (set to disable)",
)
@click.option(
    "--imdb-window-seconds",
    envvar="IMDB_WINDOW_SECONDS",
    show_envvar=True,
    type=float,
    default=1.0,
    show_default=True,
    help="Duration in seconds for the IMDb rate-limit window",
)
@click.option(
    "--imdb-queue",
    envvar="IMDB_QUEUE",
    show_envvar=True,
    type=click.Path(path_type=Path),
    default=Path("imdb_queue.json"),
    show_default=True,
    help="Path to persistent IMDb retry queue",
)
def main(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    qdrant_host: Optional[str],
    qdrant_port: int,
    qdrant_grpc_port: int,
    qdrant_https: bool,
    qdrant_prefer_grpc: bool,
    upsert_buffer_size: int,
    plex_chunk_size: int,
    enrichment_batch_size: int,
    enrichment_workers: int,
    dense_model: str,
    sparse_model: str,
    continuous: bool,
    delay: float,
    imdb_cache: Path,
    imdb_max_retries: int,
    imdb_backoff: float,
    imdb_requests_per_window: Optional[int],
    imdb_window_seconds: float,
    imdb_queue: Path,
) -> None:
    """Entry-point for the ``load-data`` script."""

    asyncio.run(
        load_media(
            plex_url,
            plex_token,
            tmdb_api_key,
            sample_dir,
            qdrant_url,
            qdrant_api_key,
            qdrant_host,
            qdrant_port,
            qdrant_grpc_port,
            qdrant_https,
            qdrant_prefer_grpc,
            dense_model,
            sparse_model,
            continuous,
            delay,
            imdb_cache,
            imdb_max_retries,
            imdb_backoff,
            imdb_requests_per_window,
            imdb_window_seconds,
            imdb_queue,
            upsert_buffer_size,
            plex_chunk_size,
            enrichment_batch_size,
            enrichment_workers,
        )
    )


async def load_media(
    plex_url: Optional[str],
    plex_token: Optional[str],
    tmdb_api_key: Optional[str],
    sample_dir: Optional[Path],
    qdrant_url: Optional[str],
    qdrant_api_key: Optional[str],
    qdrant_host: Optional[str],
    qdrant_port: int,
    qdrant_grpc_port: int,
    qdrant_https: bool,
    qdrant_prefer_grpc: bool,
    dense_model_name: str,
    sparse_model_name: str,
    continuous: bool,
    delay: float,
    imdb_cache: Path,
    imdb_max_retries: int,
    imdb_backoff: float,
    imdb_requests_per_window: Optional[int],
    imdb_window_seconds: float,
    imdb_queue: Path,
    upsert_buffer_size: int,
    plex_chunk_size: int,
    enrichment_batch_size: int,
    enrichment_workers: int,
) -> None:
    """Orchestrate one or more runs of :func:`run`."""

    while True:
        await run(
            plex_url,
            plex_token,
            tmdb_api_key,
            sample_dir,
            qdrant_url,
            qdrant_api_key,
            qdrant_host,
            qdrant_port,
            qdrant_grpc_port,
            qdrant_https,
            qdrant_prefer_grpc,
            dense_model_name,
            sparse_model_name,
            imdb_cache,
            imdb_max_retries,
            imdb_backoff,
            imdb_requests_per_window,
            imdb_window_seconds,
            imdb_queue,
            upsert_buffer_size,
            plex_chunk_size,
            enrichment_batch_size,
            enrichment_workers,
        )
        if not continuous:
            break

        await asyncio.sleep(delay)


if __name__ == "__main__":
    main()
