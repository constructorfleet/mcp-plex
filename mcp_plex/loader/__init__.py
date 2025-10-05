"""Utilities for loading Plex metadata with IMDb and TMDb details."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import warnings
from pathlib import Path
from typing import AsyncIterator, List, Optional, Sequence

import click
import httpx
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from .imdb_cache import IMDbCache
from ..common.types import (
    AggregatedItem,
    ExternalIDs,
    IMDbTitle,
    PlexItem,
    TMDBEpisode,
    TMDBItem,
    TMDBMovie,
    TMDBShow,
)
from .utils import (
    close_coroutines as _close_coroutines,
    gather_in_batches as _gather_in_batches,
    iter_gather_in_batches as _iter_gather_in_batches,
    require_positive as _require_positive,
    resolve_dense_model_params as _resolve_dense_model_params_impl,
)
from .ingestion import IngestionTask, IngestBatch
from .ingestion.utils import chunk_sequence as _chunk_sequence, load_from_sample as _load_from_sample_impl
from .enrichment import EnrichmentTask, IMDbRetryQueue
from .enrichment.utils import (
    build_plex_item as _build_plex_item_impl,
    extract_external_ids as _extract_external_ids_impl,
    fetch_imdb as _fetch_imdb_impl,
    fetch_imdb_batch as _fetch_imdb_batch_impl,
    fetch_tmdb_episode as _fetch_tmdb_episode_impl,
    fetch_tmdb_movie as _fetch_tmdb_movie_impl,
    fetch_tmdb_show as _fetch_tmdb_show_impl,
    load_imdb_retry_queue as _load_imdb_retry_queue_impl,
    persist_imdb_retry_queue as _persist_imdb_retry_queue_impl,
    process_imdb_retry_queue as _process_imdb_retry_queue_impl,
    resolve_tmdb_season_number,
)
from .storage import StorageTask
from .storage.utils import (
    build_point,
    ensure_collection as _ensure_collection_impl,
    process_qdrant_retry_queue as _process_qdrant_retry_queue_impl,
    upsert_in_batches as _upsert_in_batches_impl,
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
    message=r".*'mcp_plex\.loader' found in sys.modules after import of package 'mcp_plex'.*",
    category=RuntimeWarning,
)

_IMDbRetryQueue = IMDbRetryQueue

_imdb_cache: IMDbCache | None = None
_imdb_max_retries: int = 3
_imdb_backoff: float = 1.0
_imdb_retry_queue: IMDbRetryQueue | None = None
_imdb_batch_limit: int = 5
_qdrant_batch_size: int = 1000
_qdrant_upsert_buffer_size: int = 200
_qdrant_max_concurrent_upserts: int = 4
_qdrant_retry_attempts: int = 3
_qdrant_retry_backoff: float = 1.0


async def _fetch_imdb(
    client: httpx.AsyncClient,
    imdb_id: str,
    *,
    cache: IMDbCache | None = None,
    max_retries: int | None = None,
    backoff: float | None = None,
    retry_queue: IMDbRetryQueue | None = None,
    logger_override: logging.Logger | None = None,
) -> Optional[IMDbTitle]:
    effective_logger = logger_override or logger
    return await _fetch_imdb_impl(
        client,
        imdb_id,
        cache=cache if cache is not None else _imdb_cache,
        max_retries=max_retries if max_retries is not None else _imdb_max_retries,
        backoff=backoff if backoff is not None else _imdb_backoff,
        retry_queue=retry_queue if retry_queue is not None else _imdb_retry_queue,
        logger=effective_logger,
    )


async def _fetch_imdb_batch(
    client: httpx.AsyncClient, imdb_ids: Sequence[str]
) -> dict[str, Optional[IMDbTitle]]:
    return await _fetch_imdb_batch_impl(
        client,
        imdb_ids,
        cache=_imdb_cache,
        batch_limit=_imdb_batch_limit,
        max_retries=_imdb_max_retries,
        backoff=_imdb_backoff,
        retry_queue=_imdb_retry_queue,
        logger=logger,
    )


def _load_imdb_retry_queue(path: Path) -> None:
    global _imdb_retry_queue
    _imdb_retry_queue = _load_imdb_retry_queue_impl(path, logger)


async def _process_imdb_retry_queue(client: httpx.AsyncClient) -> None:
    if _imdb_retry_queue is None:
        return
    async def _retry_fetch(client: httpx.AsyncClient, imdb_id: str, **_: object) -> Optional[IMDbTitle]:
        return await _fetch_imdb(client, imdb_id)
    await _process_imdb_retry_queue_impl(
        client,
        _imdb_retry_queue,
        cache=_imdb_cache,
        max_retries=_imdb_max_retries,
        backoff=_imdb_backoff,
        logger=logger,
        fetch_fn=_retry_fetch,
    )


def _persist_imdb_retry_queue(path: Path) -> None:
    if _imdb_retry_queue is None:
        return
    _persist_imdb_retry_queue_impl(path, _imdb_retry_queue)




def _extract_external_ids(item: PlexPartialObject) -> ExternalIDs:
    """Extract IMDb and TMDb IDs from a Plex object."""

    return _extract_external_ids_impl(item)


def _build_plex_item(item: PlexPartialObject) -> PlexItem:
    """Convert a Plex object into the internal :class:`PlexItem`."""

    return _build_plex_item_impl(item)



async def _iter_from_plex(
    server: PlexServer, tmdb_api_key: str, *, batch_size: int = 50
) -> AsyncIterator[AggregatedItem]:
    """Yield items from a live Plex server as they are enriched."""

    async with httpx.AsyncClient(timeout=30) as client:
        movie_section = server.library.section("Movies")
        movie_keys = [int(m.ratingKey) for m in movie_section.all()]
        movies = server.fetchItems(movie_keys) if movie_keys else []
        movie_imdb_ids = [
            _extract_external_ids(m).imdb for m in movies if _extract_external_ids(m).imdb
        ]
        movie_imdb_map = (
            await _fetch_imdb_batch(client, movie_imdb_ids) if movie_imdb_ids else {}
        )

        async def _augment_movie(movie: PlexPartialObject) -> AggregatedItem:
            ids = _extract_external_ids(movie)
            imdb = movie_imdb_map.get(ids.imdb) if ids.imdb else None
            tmdb = (
                await _fetch_tmdb_movie(client, ids.tmdb, tmdb_api_key)
                if ids.tmdb
                else None
            )
            return AggregatedItem(plex=_build_plex_item(movie), imdb=imdb, tmdb=tmdb)

        movie_tasks = [_augment_movie(movie) for movie in movies]
        if movie_tasks:
            async for result in _iter_gather_in_batches(movie_tasks, batch_size):
                yield result

        show_section = server.library.section("TV Shows")
        show_keys = [int(s.ratingKey) for s in show_section.all()]
        full_shows = server.fetchItems(show_keys) if show_keys else []
        for full_show in full_shows:
            show_ids = _extract_external_ids(full_show)
            show_tmdb: Optional[TMDBShow] = None
            if show_ids.tmdb:
                show_tmdb = await _fetch_tmdb_show(client, show_ids.tmdb, tmdb_api_key)
            episode_keys = [int(e.ratingKey) for e in full_show.episodes()]
            episodes = server.fetchItems(episode_keys) if episode_keys else []
            ep_imdb_ids = [
                _extract_external_ids(e).imdb
                for e in episodes
                if _extract_external_ids(e).imdb
            ]
            ep_imdb_map = (
                await _fetch_imdb_batch(client, ep_imdb_ids) if ep_imdb_ids else {}
            )

            async def _augment_episode(episode: PlexPartialObject) -> AggregatedItem:
                ids = _extract_external_ids(episode)
                imdb = ep_imdb_map.get(ids.imdb) if ids.imdb else None
                season = resolve_tmdb_season_number(show_tmdb, episode)
                ep_num = getattr(episode, "index", None)
                tmdb_episode = (
                    await _fetch_tmdb_episode(
                        client, show_tmdb.id, season, ep_num, tmdb_api_key
                    )
                    if show_tmdb and season is not None and ep_num is not None
                    else None
                )
                tmdb: Optional[TMDBItem] = tmdb_episode or show_tmdb
                return AggregatedItem(
                    plex=_build_plex_item(episode), imdb=imdb, tmdb=tmdb
                )

            episode_tasks = [_augment_episode(ep) for ep in episodes]
            if episode_tasks:
                async for result in _iter_gather_in_batches(episode_tasks, batch_size):
                    yield result


async def _load_from_plex(
    server: PlexServer, tmdb_api_key: str, *, batch_size: int = 50
) -> List[AggregatedItem]:
    """Retain list-based API for tests by consuming :func:`_iter_from_plex`."""

    return [item async for item in _iter_from_plex(server, tmdb_api_key, batch_size=batch_size)]

async def _fetch_tmdb_movie(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBMovie]:
    return await _fetch_tmdb_movie_impl(client, tmdb_id, api_key, logger)


async def _fetch_tmdb_show(
    client: httpx.AsyncClient, tmdb_id: str, api_key: str
) -> Optional[TMDBShow]:
    return await _fetch_tmdb_show_impl(client, tmdb_id, api_key, logger)


async def _fetch_tmdb_episode(
    client: httpx.AsyncClient,
    show_id: int,
    season_number: int,
    episode_number: int,
    api_key: str,
) -> Optional[TMDBEpisode]:
    return await _fetch_tmdb_episode_impl(
        client, show_id, season_number, episode_number, api_key, logger
    )


def _load_from_sample(sample_dir: Path) -> List[AggregatedItem]:
    return _load_from_sample_impl(sample_dir)


async def _ensure_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    *,
    dense_size: int,
    dense_distance: models.Distance,
    logger_override: logging.Logger | None = None,
) -> None:
    await _ensure_collection_impl(
        client,
        collection_name,
        dense_size=dense_size,
        dense_distance=dense_distance,
        logger=logger_override or logger,
    )


async def _upsert_in_batches(
    client: AsyncQdrantClient,
    collection_name: str,
    points: Sequence[models.PointStruct],
    *,
    batch_size: int | None = None,
    retry_queue: asyncio.Queue[list[models.PointStruct]] | None = None,
    logger_override: logging.Logger | None = None,
) -> None:
    await _upsert_in_batches_impl(
        client,
        collection_name,
        points,
        batch_size=batch_size if batch_size is not None else _qdrant_batch_size,
        retry_queue=retry_queue,
        logger=logger_override or logger,
    )


async def _process_qdrant_retry_queue(
    client: AsyncQdrantClient,
    collection_name: str,
    retry_queue: asyncio.Queue[list[models.PointStruct]],
) -> None:
    await _process_qdrant_retry_queue_impl(
        client,
        collection_name,
        retry_queue,
        logger,
        max_attempts=_qdrant_retry_attempts,
        backoff=_qdrant_retry_backoff,
    )


def _resolve_dense_model_params(model_name: str) -> tuple[int, models.Distance]:
    return _resolve_dense_model_params_impl(model_name)


class LoaderPipeline:
    """Coordinate ingestion, enrichment, and Qdrant storage."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection_name: str,
        dense_model_name: str,
        sparse_model_name: str,
        tmdb_api_key: str | None,
        sample_items: list[AggregatedItem] | None,
        plex_server: PlexServer | None,
        plex_chunk_size: int,
        enrichment_batch_size: int,
        enrichment_workers: int,
        upsert_buffer_size: int,
        max_concurrent_upserts: int,
    ) -> None:
        self._client = client
        self._collection_name = collection_name
        self._dense_model_name = dense_model_name
        self._sparse_model_name = sparse_model_name
        self._tmdb_api_key = tmdb_api_key
        self._sample_items = sample_items
        self._server = plex_server
        self._plex_chunk_size = _require_positive(plex_chunk_size, name="plex_chunk_size")
        self._enrichment_batch_size = _require_positive(
            enrichment_batch_size, name="enrichment_batch_size"
        )
        self._enrichment_workers = _require_positive(
            enrichment_workers, name="enrichment_workers"
        )
        self._upsert_buffer_size = _require_positive(
            upsert_buffer_size, name="upsert_buffer_size"
        )
        self._max_concurrent_upserts = _require_positive(
            max_concurrent_upserts, name="max_concurrent_upserts"
        )

        if self._sample_items is None and self._server is None:
            raise RuntimeError("Either sample_items or plex_server must be provided")
        if self._sample_items is None and not self._tmdb_api_key:
            raise RuntimeError("TMDB API key required for live ingestion")

        self._ingest_queue: asyncio.Queue[IngestBatch | None] = asyncio.Queue(
            maxsize=self._enrichment_workers * 2
        )
        self._points_queue: asyncio.Queue[list[models.PointStruct] | None] = (
            asyncio.Queue()
        )
        self._upsert_capacity = asyncio.Semaphore(self._max_concurrent_upserts)
        self._items: list[AggregatedItem] = []
        self._storage_task: StorageTask | None = None
        self._enrichment_task: EnrichmentTask | None = None
        self._dense_params: tuple[int, models.Distance] | None = None
        self._collection_ensured = False

    @property
    def qdrant_retry_queue(self) -> asyncio.Queue[list[models.PointStruct]]:
        if self._storage_task is None:
            raise RuntimeError("Pipeline has not been executed")
        return self._storage_task.retry_queue

    @property
    def items(self) -> list[AggregatedItem]:
        return self._items

    async def ensure_collection(self) -> None:
        if self._dense_params is None:
            self._dense_params = _resolve_dense_model_params(self._dense_model_name)
        dense_size, dense_distance = self._dense_params
        if not hasattr(self._client, "collection_exists"):
            self._collection_ensured = True
            return
        await _ensure_collection(
            self._client,
            self._collection_name,
            dense_size=dense_size,
            dense_distance=dense_distance,
        )
        self._collection_ensured = True

    def _log_progress(self, stage: str, count: int, start: float, queue_size: int) -> None:
        elapsed = time.perf_counter() - start
        rate = count / elapsed if elapsed > 0 else 0.0
        logger.info(
            "%s processed %d items (%.2f items/sec, queue size=%d)",
            stage,
            count,
            rate,
            queue_size,
        )

    async def execute(self) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            imdb_queue = _imdb_retry_queue or IMDbRetryQueue()
            if _imdb_retry_queue is None:
                globals()['_imdb_retry_queue'] = imdb_queue
            ingestion = IngestionTask(
                self._ingest_queue,
                sample_items=self._sample_items,
                plex_server=self._server,
                plex_chunk_size=self._plex_chunk_size,
                enrichment_batch_size=self._enrichment_batch_size,
                enrichment_workers=self._enrichment_workers,
                log_progress=self._log_progress,
            )
            enrichment = EnrichmentTask(
                self._ingest_queue,
                self._points_queue,
                http_client=client,
                tmdb_api_key=self._tmdb_api_key,
                imdb_cache=_imdb_cache,
                imdb_retry_queue=imdb_queue,
                imdb_batch_limit=_imdb_batch_limit,
                imdb_max_retries=_imdb_max_retries,
                imdb_backoff=_imdb_backoff,
                dense_model_name=self._dense_model_name,
                sparse_model_name=self._sparse_model_name,
                enrichment_batch_size=self._enrichment_batch_size,
                worker_count=self._enrichment_workers,
                upsert_buffer_size=self._upsert_buffer_size,
                upsert_capacity=self._upsert_capacity,
                log_progress=self._log_progress,
                logger=logger,
            )
            if self._dense_params is None:
                self._dense_params = _resolve_dense_model_params(
                    self._dense_model_name
                )
            dense_size, dense_distance = self._dense_params
            storage = StorageTask(
                self._points_queue,
                client=self._client,
                collection_name=self._collection_name,
                dense_size=dense_size,
                dense_distance=dense_distance,
                upsert_batch_size=_qdrant_batch_size,
                worker_count=self._max_concurrent_upserts,
                upsert_capacity=self._upsert_capacity,
                log_progress=self._log_progress,
                logger=logger,
                retry_attempts=_qdrant_retry_attempts,
                retry_backoff=_qdrant_retry_backoff,
                ensure_collection_fn=_ensure_collection,
                upsert_fn=_upsert_in_batches,
            )
            if not self._collection_ensured:
                if hasattr(self._client, "collection_exists"):
                    await storage.ensure_collection()
                self._collection_ensured = True

            ingest_task = asyncio.create_task(ingestion.run())
            enrichment_tasks = enrichment.start_workers()
            storage_tasks = storage.start_workers()

            error: BaseException | None = None
            try:
                await ingest_task
                await self._ingest_queue.join()
                await asyncio.gather(*enrichment_tasks)
                await self._points_queue.join()
            except BaseException as exc:
                error = exc
            finally:
                for _ in range(self._max_concurrent_upserts):
                    await self._points_queue.put(None)
                storage_results = await asyncio.gather(
                    *storage_tasks, return_exceptions=True
                )
                for result in storage_results:
                    if isinstance(result, BaseException) and not isinstance(
                        result, asyncio.CancelledError
                    ):
                        if error is None:
                            error = result
                        break
                for task in enrichment_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*enrichment_tasks, return_exceptions=True)
                if not ingest_task.done():
                    ingest_task.cancel()
                    await asyncio.gather(ingest_task, return_exceptions=True)
                if error is not None:
                    raise error

            self._items = enrichment.items
            self._storage_task = storage
            self._enrichment_task = enrichment

    async def drain_retry_queue(self) -> None:
        if self._storage_task is None:
            return
        await self._storage_task.drain_retry_queue()


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
    upsert_buffer_size: int = _qdrant_upsert_buffer_size,
    plex_chunk_size: int = 200,
    enrichment_batch_size: int = 100,
    enrichment_workers: int = 4,
) -> None:
    global _imdb_cache, _imdb_max_retries, _imdb_backoff, _imdb_retry_queue
    _imdb_cache = IMDbCache(imdb_cache_path) if imdb_cache_path else None
    _imdb_max_retries = imdb_max_retries
    _imdb_backoff = imdb_backoff
    if imdb_queue_path:
        _load_imdb_retry_queue(imdb_queue_path)
        async with httpx.AsyncClient(timeout=30) as client:
            await _process_imdb_retry_queue(client)
    else:
        _imdb_retry_queue = IMDbRetryQueue()

    _require_positive(upsert_buffer_size, name="upsert_buffer_size")
    _require_positive(plex_chunk_size, name="plex_chunk_size")
    _require_positive(enrichment_batch_size, name="enrichment_batch_size")
    _require_positive(enrichment_workers, name="enrichment_workers")

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

    items: List[AggregatedItem]
    if sample_dir is not None:
        logger.info("Loading sample data from %s", sample_dir)
        sample_items = _load_from_sample(sample_dir)
        pipeline = LoaderPipeline(
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
        )
    else:
        if PlexServer is None:
            raise RuntimeError("plexapi is required for live loading")
        if not plex_url or not plex_token:
            raise RuntimeError("PLEX_URL and PLEX_TOKEN must be provided")
        if not tmdb_api_key:
            raise RuntimeError("TMDB_API_KEY must be provided")
        logger.info("Loading data from Plex server %s", plex_url)
        server = PlexServer(plex_url, plex_token)
        pipeline = LoaderPipeline(
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

    await pipeline.ensure_collection()
    await pipeline.execute()
    items = pipeline.items
    logger.info("Loaded %d items", len(items))
    if not items:
        logger.info("No points to upsert")

    await pipeline.drain_retry_queue()

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
    help="Number of media items to buffer before scheduling an async storage write",
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
    imdb_queue: Path,
) -> None:
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
    imdb_queue: Path,
    upsert_buffer_size: int,
    plex_chunk_size: int,
    enrichment_batch_size: int,
    enrichment_workers: int,
) -> None:
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


__all__ = [
    "LoaderPipeline",
    "_build_plex_item",
    "_chunk_sequence",
    "_close_coroutines",
    "_ensure_collection",
    "_IMDbRetryQueue",
    "_extract_external_ids",
    "_fetch_imdb",
    "_fetch_imdb_batch",
    "_fetch_tmdb_episode",
    "_fetch_tmdb_movie",
    "_fetch_tmdb_show",
    "_gather_in_batches",
    "_iter_gather_in_batches",
    "_load_from_sample",
    "_load_imdb_retry_queue",
    "_persist_imdb_retry_queue",
    "_process_imdb_retry_queue",
    "_process_qdrant_retry_queue",
    "_resolve_dense_model_params",
    "_require_positive",
    "_upsert_in_batches",
    "build_point",
    "load_media",
    "main",
    "resolve_tmdb_season_number",
    "run",
]
