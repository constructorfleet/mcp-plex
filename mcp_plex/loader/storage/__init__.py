from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable

from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

from ..utils import require_positive
from .types import StorageBatch
from .utils import (
    build_point,
    ensure_collection,
    process_qdrant_retry_queue,
    upsert_in_batches,
)


class StorageTask:
    """Persist enriched points into Qdrant."""

    def __init__(
        self,
        points_queue: asyncio.Queue[StorageBatch | None],
        *,
        client: AsyncQdrantClient,
        collection_name: str,
        dense_size: int,
        dense_distance: models.Distance,
        upsert_batch_size: int,
        worker_count: int,
        upsert_capacity: asyncio.Semaphore,
        log_progress: Callable[[str, int, float, int], None],
        logger: logging.Logger,
        retry_attempts: int,
        retry_backoff: float,
        ensure_collection_fn: Callable[..., Awaitable[None]] = ensure_collection,
        upsert_fn: Callable[..., Awaitable[None]] = upsert_in_batches,
    ) -> None:
        self._points_queue = points_queue
        self._client = client
        self._collection_name = collection_name
        self._upsert_batch_size = require_positive(upsert_batch_size, name="upsert_batch_size")
        self._worker_count = require_positive(worker_count, name="worker_count")
        self._upsert_capacity = upsert_capacity
        self._log_progress = log_progress
        self._logger = logger
        self._retry_attempts = retry_attempts
        self._retry_backoff = retry_backoff
        self._ensure_collection_fn = ensure_collection_fn
        self._upsert_fn = upsert_fn

        self._upserted_points = 0
        self._upsert_start = time.perf_counter()
        self._retry_queue: asyncio.Queue[list[models.PointStruct]] = asyncio.Queue()
        self._dense_size = dense_size
        self._dense_distance = dense_distance

    @property
    def retry_queue(self) -> asyncio.Queue[list[models.PointStruct]]:
        return self._retry_queue

    async def ensure_collection(self) -> None:
        await self._ensure_collection_fn(
            self._client,
            self._collection_name,
            dense_size=self._dense_size,
            dense_distance=self._dense_distance,
            logger_override=self._logger,
        )

    def start_workers(self) -> list[asyncio.Task[None]]:
        return [
            asyncio.create_task(self._worker(worker_id))
            for worker_id in range(self._worker_count)
        ]

    async def _worker(self, worker_id: int) -> None:
        while True:
            batch = await self._points_queue.get()
            if batch is None:
                self._points_queue.task_done()
                break
            self._logger.info(
                "Storage worker %d handling %d points (queue size=%d)",
                worker_id,
                len(batch),
                self._points_queue.qsize(),
            )
            try:
                await self._upsert_fn(
                    self._client,
                    self._collection_name,
                    batch,
                    batch_size=self._upsert_batch_size,
                    retry_queue=self._retry_queue,
                    logger_override=self._logger,
                )
                if self._upserted_points == 0:
                    self._upsert_start = time.perf_counter()
                self._upserted_points += len(batch)
                self._log_progress(
                    f"Storage worker {worker_id}",
                    self._upserted_points,
                    self._upsert_start,
                    self._points_queue.qsize(),
                )
            finally:
                self._points_queue.task_done()
                self._upsert_capacity.release()

    async def drain_retry_queue(self) -> None:
        await process_qdrant_retry_queue(
            self._client,
            self._collection_name,
            self._retry_queue,
            self._logger,
            max_attempts=self._retry_attempts,
            backoff=self._retry_backoff,
        )


__all__ = ["StorageTask", "build_point", "ensure_collection", "process_qdrant_retry_queue"]
