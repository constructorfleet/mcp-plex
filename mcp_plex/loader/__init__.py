"""Loader orchestration utilities and staged pipeline helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, TypedDict, cast

import httpx
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from plexapi.base import PlexPartialObject as _PlexPartialObject
from plexapi.server import PlexServer

from .imdb_cache import IMDbCache
from .pipeline.channels import (
    IMDbRetryQueue,
    INGEST_DONE,
    IngestQueue,
    PersistenceQueue,
    chunk_sequence,
)
from ..common.validation import require_positive
from .pipeline.orchestrator import LoaderOrchestrator
from .pipeline.persistence import PersistenceStage as _PersistenceStage
from ..common.types import (
    AggregatedItem,
    ExternalIDs,
    IMDbTitle,
    JSONValue,
)
from . import qdrant as _qdrant
from . import samples as samples
from .samples import _load_from_sample as _load_from_sample

_DENSE_MODEL_PARAMS = _qdrant._DENSE_MODEL_PARAMS
_resolve_dense_model_params = _qdrant._resolve_dense_model_params
_is_local_qdrant = _qdrant._is_local_qdrant
_ensure_collection = _qdrant._ensure_collection
_build_point_text = _qdrant._build_point_text
_build_point_payload = _qdrant._build_point_payload
QdrantPayload = _qdrant.QdrantPayload
build_point = _qdrant.build_point
_upsert_in_batches = _qdrant._upsert_in_batches
_process_qdrant_retry_queue = _qdrant._process_qdrant_retry_queue
_point_id_from_rating_key = _qdrant._point_id_from_rating_key
_normalise_point_id = _qdrant._normalise_point_id
_delete_missing_rating_keys = _qdrant._delete_missing_rating_keys
_existing_point_ids = _qdrant._existing_point_ids
_find_record_by_external_ids = _qdrant._find_record_by_external_ids
rehydrate_aggregated_item = _qdrant.rehydrate_aggregated_item

PlexPartialObject = _PlexPartialObject

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*'mcp_plex\\.loader' found in sys.modules after import of package 'mcp_plex'.*",
    category=RuntimeWarning,
)

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .pipeline.enrichment import _RequestThrottler

IMDB_BATCH_LIMIT: int = 5
DEFAULT_QDRANT_BATCH_SIZE: int = 1000
DEFAULT_QDRANT_UPSERT_BUFFER_SIZE: int = 200
DEFAULT_QDRANT_MAX_CONCURRENT_UPSERTS: int = 4
DEFAULT_QDRANT_RETRY_ATTEMPTS: int = 3
DEFAULT_QDRANT_RETRY_BACKOFF: float = 1.0
MIN_CONTINUOUS_SLEEP: float = 0.1
INGEST_QUEUE_MULTIPLIER: int = 2
PERSISTENCE_QUEUE_MULTIPLIER: int = 2


def _queue_capacity(worker_count: int, multiplier: int) -> int:
    """Derive a positive queue capacity based on worker parallelism."""

    return max(1, int(worker_count) * multiplier)


@dataclass(slots=True)
class IMDbRuntimeConfig:
    """Runtime configuration for IMDb enrichment helpers."""

    cache: IMDbCache | None
    max_retries: int
    backoff: float
    retry_queue: IMDbRetryQueue
    requests_per_window: int | None
    window_seconds: float
    _throttle: _RequestThrottler | None = field(default=None, init=False, repr=False)

    def get_throttle(self) -> _RequestThrottler | None:
        """Return the shared rate limiter, creating it on first use."""

        if self.requests_per_window is None:
            return None
        if self._throttle is None:
            from .pipeline import enrichment as enrichment_mod

            self._throttle = enrichment_mod._RequestThrottler(
                limit=self.requests_per_window,
                interval=float(self.window_seconds),
            )
        return self._throttle


@dataclass(slots=True)
class QdrantRuntimeConfig:
    """Runtime configuration for Qdrant persistence helpers."""

    batch_size: int = DEFAULT_QDRANT_BATCH_SIZE
    retry_attempts: int = DEFAULT_QDRANT_RETRY_ATTEMPTS
    retry_backoff: float = DEFAULT_QDRANT_RETRY_BACKOFF


async def _fetch_imdb(
    client: httpx.AsyncClient,
    imdb_id: str,
    config: IMDbRuntimeConfig,
) -> IMDbTitle | None:
    """Fetch metadata for an IMDb ID with caching, retry, and throttling."""

    from .pipeline import enrichment as enrichment_mod

    return await enrichment_mod._fetch_imdb(
        client,
        imdb_id,
        cache=config.cache,
        throttle=config.get_throttle(),
        max_retries=config.max_retries,
        backoff=config.backoff,
        retry_queue=config.retry_queue,
    )


def _load_imdb_retry_queue(path: Path) -> IMDbRetryQueue:
    """Populate the retry queue from a JSON file if it exists."""

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
    return IMDbRetryQueue(ids)


async def _process_imdb_retry_queue(
    client: httpx.AsyncClient,
    config: IMDbRuntimeConfig,
) -> None:
    """Attempt to fetch queued IMDb IDs, re-queueing failures."""

    if config.retry_queue.empty():
        return
    size = config.retry_queue.qsize()
    for _ in range(size):
        imdb_id = await config.retry_queue.get()
        title = await _fetch_imdb(client, imdb_id, config)
        if title is None:
            await config.retry_queue.put(imdb_id)


def _persist_imdb_retry_queue(path: Path, queue: IMDbRetryQueue) -> None:
    """Persist the retry queue to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue.snapshot()))


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
    imdb_config: IMDbRuntimeConfig,
    qdrant_config: QdrantRuntimeConfig,
) -> tuple[
    LoaderOrchestrator,
    list[AggregatedItem],
    asyncio.Queue[list[models.PointStruct]],
    set[str],
]:
    """Wire the staged loader pipeline and return the orchestrator helpers."""
    from .pipeline.ingestion import IngestionStage
    from .pipeline.enrichment import EnrichmentStage

    rating_keys_seen: set[str] = set()

    ingest_queue_capacity = _queue_capacity(
        enrichment_workers, INGEST_QUEUE_MULTIPLIER
    )
    persistence_queue_capacity = _queue_capacity(
        max_concurrent_upserts, PERSISTENCE_QUEUE_MULTIPLIER
    )

    ingest_queue: IngestQueue = IngestQueue(maxsize=ingest_queue_capacity)
    persistence_queue: PersistenceQueue = PersistenceQueue(
        maxsize=persistence_queue_capacity
    )
    imdb_queue = imdb_config.retry_queue

    upsert_capacity = asyncio.Semaphore(max_concurrent_upserts)
    qdrant_retry_queue: asyncio.Queue[list[models.PointStruct]] = asyncio.Queue()
    items: list[AggregatedItem] = []
    upserted = 0
    upsert_start = time.perf_counter()

    async def _filter_new_items(
        batch: Sequence[AggregatedItem | models.PointStruct],
    ) -> tuple[list[AggregatedItem], int]:
        if not batch:
            return [], 0

        # Filter out PointStruct instances
        filtered_batch = [item for item in batch if isinstance(item, AggregatedItem)]

        point_pairs: list[tuple[AggregatedItem, int | str]] = []
        for item in filtered_batch:
            rating_key = str(getattr(item.plex, "rating_key", ""))
            if rating_key:
                rating_keys_seen.add(rating_key)
            point_pairs.append((item, _point_id_from_rating_key(item.plex.rating_key)))

        existing_ids = await _existing_point_ids(
            client, collection_name, [point_id for _, point_id in point_pairs]
        )

        new_items: list[AggregatedItem] = []
        skipped = 0
        for item, point_id in point_pairs:
            if _normalise_point_id(point_id) in existing_ids:
                logger.debug(
                    "Skipping Plex rating key %s; already indexed in Qdrant.",
                    item.plex.rating_key,
                )
                skipped += 1
                continue
            new_items.append(item)
        return new_items, skipped

    async def _upsert_aggregated(
        batch: Sequence[AggregatedItem | models.PointStruct],
    ) -> None:
        if not batch:
            return
        if isinstance(batch[0], models.PointStruct):
            points = [point for point in batch if isinstance(point, models.PointStruct)]
            for point_chunk in chunk_sequence(points, upsert_buffer_size):
                await _upsert_in_batches(
                    client,
                    collection_name,
                    point_chunk,
                    batch_size=qdrant_config.batch_size,
                    retry_queue=qdrant_retry_queue,
                )
            return

        filtered_items, skipped = await _filter_new_items(batch)

        if not filtered_items:
            logger.info(
                "Skipping batch of %d item(s); already present in Qdrant.",
                len(batch),
            )
            return

        if skipped:
            logger.info(
                "Skipped %d existing item(s); %d new item(s) remain.",
                skipped,
                len(filtered_items),
            )

        items.extend(filtered_items)
        points = [
            build_point(item, dense_model_name, sparse_model_name)
            for item in filtered_items
        ]
        for point_chunk in chunk_sequence(points, upsert_buffer_size):
            await _upsert_in_batches(
                client,
                collection_name,
                point_chunk,
                batch_size=qdrant_config.batch_size,
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

    async def _lookup_existing_payload(
        external_ids: ExternalIDs,
        plex_guid: str | None,
    ) -> dict[str, JSONValue] | None:
        record = await _find_record_by_external_ids(
            client,
            collection_name,
            imdb_id=external_ids.imdb,
            tmdb_id=external_ids.tmdb,
            plex_guid=plex_guid,
        )
        if record is None:
            return None
        payload = record.payload
        if isinstance(payload, dict):
            return cast(dict[str, JSONValue], payload)
        return None

    ingestion_stage = IngestionStage(
        plex_server=plex_server,
        sample_items=sample_items,
        movie_batch_size=plex_chunk_size,
        episode_batch_size=plex_chunk_size,
        season_batch_size=plex_chunk_size,
        show_batch_size=plex_chunk_size,
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
        imdb_cache=imdb_config.cache,
        imdb_max_retries=imdb_config.max_retries,
        imdb_backoff=imdb_config.backoff,
        imdb_batch_limit=IMDB_BATCH_LIMIT,
        imdb_requests_per_window=imdb_config.requests_per_window,
        imdb_window_seconds=imdb_config.window_seconds,
        existing_payload_lookup=_lookup_existing_payload,
    )

    persistence_stage = _PersistenceStage(
        client=client, # type: ignore
        collection_name=collection_name,
        dense_vector_name=dense_model_name,
        sparse_vector_name=sparse_model_name,
        persistence_queue=persistence_queue,
        retry_queue=qdrant_retry_queue, # type: ignore
        upsert_semaphore=upsert_capacity,
        upsert_buffer_size=upsert_buffer_size,
        upsert_fn=_upsert_aggregated, # type: ignore
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

    return orchestrator, items, persistence_stage.retry_queue, rating_keys_seen # type: ignore


async def run(
    plex_url: str | None,
    plex_token: str | None,
    tmdb_api_key: str | None,
    sample_dir: Path | None,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    qdrant_host: str | None = None,
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
    upsert_buffer_size: int = DEFAULT_QDRANT_UPSERT_BUFFER_SIZE,
    plex_chunk_size: int = 200,
    enrichment_batch_size: int = 100,
    enrichment_workers: int = 4,
    qdrant_batch_size: int = DEFAULT_QDRANT_BATCH_SIZE,
    max_concurrent_upserts: int = DEFAULT_QDRANT_MAX_CONCURRENT_UPSERTS,
    qdrant_retry_attempts: int = DEFAULT_QDRANT_RETRY_ATTEMPTS,
    qdrant_retry_backoff: float = DEFAULT_QDRANT_RETRY_BACKOFF,
) -> None:
    """Core execution logic for the CLI."""

    imdb_cache = IMDbCache(imdb_cache_path) if imdb_cache_path else None
    if imdb_requests_per_window is not None:
        require_positive(imdb_requests_per_window, name="imdb_requests_per_window")
        if imdb_window_seconds <= 0:
            raise ValueError("imdb_window_seconds must be positive")
    if qdrant_retry_backoff <= 0:
        raise ValueError("qdrant_retry_backoff must be positive")

    require_positive(upsert_buffer_size, name="upsert_buffer_size")
    require_positive(plex_chunk_size, name="plex_chunk_size")
    require_positive(enrichment_batch_size, name="enrichment_batch_size")
    require_positive(enrichment_workers, name="enrichment_workers")
    require_positive(qdrant_batch_size, name="qdrant_batch_size")
    require_positive(max_concurrent_upserts, name="max_concurrent_upserts")
    require_positive(qdrant_retry_attempts, name="qdrant_retry_attempts")

    imdb_retry_queue = (
        _load_imdb_retry_queue(imdb_queue_path) if imdb_queue_path else IMDbRetryQueue()
    )
    imdb_config = IMDbRuntimeConfig(
        cache=imdb_cache,
        max_retries=imdb_max_retries,
        backoff=imdb_backoff,
        retry_queue=imdb_retry_queue,
        requests_per_window=imdb_requests_per_window,
        window_seconds=imdb_window_seconds,
    )

    if imdb_queue_path:
        async with httpx.AsyncClient(timeout=30) as client:
            await _process_imdb_retry_queue(client, imdb_config)

    qdrant_config = QdrantRuntimeConfig(
        batch_size=qdrant_batch_size,
        retry_attempts=qdrant_retry_attempts,
        retry_backoff=qdrant_retry_backoff,
    )

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
    try:
        collection_name = "media-items"
        await _ensure_collection(
            client,
            collection_name,
            dense_size=dense_size,
            dense_distance=dense_distance,
        )

        cleanup_enabled = sample_dir is None
        observed_rating_keys: set[str] = set()

        items: list[AggregatedItem]
        if sample_dir is not None:
            logger.info("Loading sample data from %s", sample_dir)
            sample_items = samples._load_from_sample(sample_dir)
            (
                orchestrator,
                items,
                qdrant_retry_queue,
                observed_rating_keys,
            ) = _build_loader_orchestrator(
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
                max_concurrent_upserts=max_concurrent_upserts,
                imdb_config=IMDbRuntimeConfig(
                    cache=imdb_config.cache,
                    max_retries=imdb_config.max_retries,
                    backoff=imdb_config.backoff,
                    retry_queue=IMDbRetryQueue(),
                    requests_per_window=imdb_config.requests_per_window,
                    window_seconds=imdb_config.window_seconds,
                ),
                qdrant_config=qdrant_config,
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
            (
                orchestrator,
                items,
                qdrant_retry_queue,
                observed_rating_keys,
            ) = _build_loader_orchestrator(
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
                max_concurrent_upserts=max_concurrent_upserts,
                imdb_config=imdb_config,
                qdrant_config=qdrant_config,
            )
            logger.info("Starting staged loader (Plex mode)")
            await orchestrator.run()
        logger.info("Loaded %d items", len(items))
        if not items:
            logger.info("No points to upsert")

        succeeded_points, failed_points = await _process_qdrant_retry_queue(
            client,
            collection_name,
            qdrant_retry_queue,
            config=qdrant_config,
        )
        retry_summary = {
            "event": "qdrant_retry_summary",
            "succeeded_points": succeeded_points,
            "failed_points": failed_points,
        }
        logger.info(
            "Qdrant retry summary: %d succeeded, %d failed",
            succeeded_points,
            failed_points,
            extra=retry_summary,
        )

        if cleanup_enabled:
            if observed_rating_keys:
                deleted_points, scanned_points = await _delete_missing_rating_keys(
                    client,
                    collection_name,
                    observed_rating_keys,
                )
                logger.info(
                    "Qdrant cleanup removed %d stale point(s) after scanning %d total.",
                    deleted_points,
                    scanned_points,
                )
            else:
                logger.info(
                    "Skipping Qdrant cleanup because no Plex rating keys were observed.",
                )
        else:
            logger.info("Skipping Qdrant cleanup in sample mode.")

        if imdb_queue_path:
            _persist_imdb_retry_queue(imdb_queue_path, imdb_config.retry_queue)

        json.dump(
            [item.model_dump(mode="json") for item in items], fp=sys.stdout, indent=2
        )
        sys.stdout.write("\n")
    finally:
        await client.close()


async def load_media(
    plex_url: str | None,
    plex_token: str | None,
    tmdb_api_key: str | None,
    sample_dir: Path | None,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    qdrant_host: str | None,
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
    imdb_requests_per_window: int | None,
    imdb_window_seconds: float,
    imdb_queue: Path,
    upsert_buffer_size: int,
    plex_chunk_size: int,
    enrichment_batch_size: int,
    enrichment_workers: int,
    qdrant_batch_size: int = DEFAULT_QDRANT_BATCH_SIZE,
    max_concurrent_upserts: int = DEFAULT_QDRANT_MAX_CONCURRENT_UPSERTS,
    qdrant_retry_attempts: int = DEFAULT_QDRANT_RETRY_ATTEMPTS,
    qdrant_retry_backoff: float = DEFAULT_QDRANT_RETRY_BACKOFF,
) -> None:
    """Orchestrate one or more runs of :func:`run`."""

    if delay < 0:
        raise ValueError(f"Delay between runs must be non-negative; received {delay!r}")

    while True:
        await run(
            plex_url=plex_url,
            plex_token=plex_token,
            tmdb_api_key=tmdb_api_key,
            sample_dir=sample_dir,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_grpc_port=qdrant_grpc_port,
            qdrant_https=qdrant_https,
            qdrant_prefer_grpc=qdrant_prefer_grpc,
            dense_model_name=dense_model_name,
            sparse_model_name=sparse_model_name,
            imdb_cache_path=imdb_cache,
            imdb_max_retries=imdb_max_retries,
            imdb_backoff=imdb_backoff,
            imdb_queue_path=imdb_queue,
            imdb_requests_per_window=imdb_requests_per_window,
            imdb_window_seconds=imdb_window_seconds,
            upsert_buffer_size=upsert_buffer_size,
            plex_chunk_size=plex_chunk_size,
            enrichment_batch_size=enrichment_batch_size,
            enrichment_workers=enrichment_workers,
            qdrant_batch_size=qdrant_batch_size,
            max_concurrent_upserts=max_concurrent_upserts,
            qdrant_retry_attempts=qdrant_retry_attempts,
            qdrant_retry_backoff=qdrant_retry_backoff,
        )
        if not continuous:
            break

        sleep_interval = delay if delay > 0 else MIN_CONTINUOUS_SLEEP
        if delay <= 0:
            logger.warning(
                "Continuous mode requested non-positive delay %s; using %s seconds instead",
                delay,
                sleep_interval,
            )
        await asyncio.sleep(sleep_interval)


class DataSourcePayload(TypedDict):
    """Structured data returned by :class:`DataSource` implementations."""

    source: str
    items: list[dict[str, JSONValue]]


def _validate_source_payload(data: DataSourcePayload, source: str) -> bool:
    if data.get("source") != source:
        return False
    items = data.get("items")
    if not isinstance(items, list):
        return False
    return all(isinstance(item, dict) for item in items)


class DataSource(ABC):
    """Abstract base class for loader data sources.

    Implementations standardize how external metadata systems return raw payloads.
    Each source must return a dictionary containing a ``source`` label and a list
    of item payloads so downstream stages can reason about provenance and shape.
    """

    @abstractmethod
    async def fetch_data(self) -> DataSourcePayload:
        """Fetch data from the source.

        Returns:
            A mapping with ``source`` set to the source identifier and ``items``
            containing a list of payload dictionaries for each fetched item.
        """

    @abstractmethod
    def validate(self, data: DataSourcePayload) -> bool:
        """Validate the fetched data.

        Args:
            data: The payload returned from :meth:`fetch_data`.

        Returns:
            ``True`` when the payload matches the expected structure for the
            source; otherwise ``False``.
        """


class PlexSource(DataSource):
    async def fetch_data(self) -> DataSourcePayload:
        """Fetch Plex metadata payloads."""

        return {"source": "plex", "items": []}

    def validate(self, data: DataSourcePayload) -> bool:
        """Validate Plex payload structure."""

        return _validate_source_payload(data, "plex")


class TMDBSource(DataSource):
    async def fetch_data(self) -> DataSourcePayload:
        """Fetch TMDb metadata payloads."""

        return {"source": "tmdb", "items": []}

    def validate(self, data: DataSourcePayload) -> bool:
        """Validate TMDb payload structure."""

        return _validate_source_payload(data, "tmdb")


class IMDbSource(DataSource):
    async def fetch_data(self) -> DataSourcePayload:
        """Fetch IMDb metadata payloads."""

        return {"source": "imdb", "items": []}

    def validate(self, data: DataSourcePayload) -> bool:
        """Validate IMDb payload structure."""

        return _validate_source_payload(data, "imdb")
