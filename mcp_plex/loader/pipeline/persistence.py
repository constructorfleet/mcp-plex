"""Persistence stage placeholder used by the loader pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Sequence, TypeAlias

from .channels import (
    PERSIST_DONE,
    PersistenceQueue,
    chunk_sequence,
    enqueue_nowait,
)
from ...common.validation import require_positive

try:  # pragma: no cover - allow import to fail when qdrant_client is absent
    from qdrant_client import AsyncQdrantClient, models
except ModuleNotFoundError:  # pragma: no cover - tooling without qdrant installed
    class AsyncQdrantClient:  # type: ignore[too-few-public-methods]
        """Fallback stub used when qdrant_client is unavailable."""

        pass

    class _ModelsStub:  # type: ignore[too-few-public-methods]
        class PointStruct:  # type: ignore[too-few-public-methods]
            ...

    models = _ModelsStub()  # type: ignore[assignment]


PersistencePayload: TypeAlias = list["models.PointStruct"]


class PersistenceStage:
    """Drain the persistence queue and coordinate Qdrant upserts."""

    def __init__(
        self,
        *,
        client: AsyncQdrantClient,
        collection_name: str,
        dense_vector_name: str,
        sparse_vector_name: str,
        persistence_queue: PersistenceQueue,
        retry_queue: asyncio.Queue[PersistencePayload],
        upsert_semaphore: asyncio.Semaphore,
        upsert_buffer_size: int,
        upsert_fn: Callable[[PersistencePayload], Awaitable[None]],
        on_batch_complete: Callable[[int, int, int], None] | None = None,
        worker_count: int = 1,
    ) -> None:
        self._client = client
        self._collection_name = str(collection_name)
        self._dense_vector_name = str(dense_vector_name)
        self._sparse_vector_name = str(sparse_vector_name)
        self._persistence_queue = persistence_queue
        self._retry_queue = retry_queue
        self._upsert_semaphore = upsert_semaphore
        self._upsert_buffer_size = require_positive(
            upsert_buffer_size, name="upsert_buffer_size"
        )
        self._upsert_fn = upsert_fn
        self._on_batch_complete = on_batch_complete
        self._logger = logging.getLogger("mcp_plex.loader.persistence")
        self._retry_flush_attempted = False
        self._worker_count = require_positive(worker_count, name="worker_count")
        self._shutdown_tokens_seen = 0

    @property
    def logger(self) -> logging.Logger:
        """Logger used by the persistence stage."""

        return self._logger

    @property
    def qdrant_client(self) -> AsyncQdrantClient:
        """Return the Qdrant client used for persistence."""

        return self._client

    @property
    def collection_name(self) -> str:
        """Name of the Qdrant collection targeted by persistence."""

        return self._collection_name

    @property
    def dense_vector_name(self) -> str:
        """Name of the dense vector configuration in the collection."""

        return self._dense_vector_name

    @property
    def sparse_vector_name(self) -> str:
        """Name of the sparse vector configuration in the collection."""

        return self._sparse_vector_name

    @property
    def persistence_queue(self) -> PersistenceQueue:
        """Queue providing batches destined for Qdrant."""

        return self._persistence_queue

    @property
    def retry_queue(self) -> asyncio.Queue[PersistencePayload]:
        """Queue used to persist batches that require retries."""

        return self._retry_queue

    @property
    def upsert_semaphore(self) -> asyncio.Semaphore:
        """Semaphore limiting concurrent Qdrant upserts."""

        return self._upsert_semaphore

    @property
    def upsert_buffer_size(self) -> int:
        """Maximum number of points per persistence batch."""

        return self._upsert_buffer_size

    async def _flush_retry_queue(self) -> int:
        """Re-enqueue retry batches so they are persisted before shutdown."""

        drained_count = 0
        while True:
            try:
                retry_payload = self._retry_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            drained_count += 1
            try:
                await self.enqueue_points(retry_payload)
            finally:
                self._retry_queue.task_done()

        if drained_count:
            self._logger.debug(
                "Re-enqueued %d retry batch(es) before persistence shutdown.",
                drained_count,
            )

        return drained_count

    def _drain_additional_sentinels(self) -> int:
        """Remove queued sentinel tokens so payloads run before shutdown."""

        drained = 0
        while True:
            try:
                queued_item = self._persistence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            if queued_item in (None, PERSIST_DONE):
                drained += 1
                self._persistence_queue.task_done()
                continue

            # Non-sentinel payload encountered; put it back and stop draining.
            self._persistence_queue.put_nowait(queued_item)
            break

        return drained

    async def enqueue_points(
        self, points: Sequence["models.PointStruct"]
    ) -> None:
        """Chunk *points* and place them on the persistence queue."""

        if not points:
            return

        total_points = len(points)
        self._logger.info(
            "Received %d point(s) for persistence; chunking with buffer size=%d.",
            total_points,
            self._upsert_buffer_size,
        )
        for chunk in chunk_sequence(list(points), self._upsert_buffer_size):
            batch = list(chunk)
            if not batch:
                continue
            await self._upsert_semaphore.acquire()
            try:
                await enqueue_nowait(self._persistence_queue, batch)
            except BaseException:
                self._upsert_semaphore.release()
                raise
            else:
                self._logger.debug(
                    "Queued persistence batch with %d point(s) (queue size=%d).",
                    len(batch),
                    self._persistence_queue.qsize(),
                )

    async def run(self, worker_id: int) -> None:
        """Drain the persistence queue until a sentinel is received."""

        self._logger.info(
            "Starting persistence worker %d (buffer size=%d).",
            worker_id,
            self._upsert_buffer_size,
        )
        while True:
            payload = await self._persistence_queue.get()
            try:
                if payload is None or payload is PERSIST_DONE:
                    sentinel_budget = 1 + self._drain_additional_sentinels()
                    drained_retry = 0
                    if not self._retry_flush_attempted:
                        drained_retry = await self._flush_retry_queue()
                        if drained_retry:
                            self._retry_flush_attempted = True
                            for _ in range(sentinel_budget):
                                await enqueue_nowait(
                                    self._persistence_queue, PERSIST_DONE
                                )
                            continue

                    drained_sentinels = max(sentinel_budget - 1, 0)
                    for _ in range(drained_sentinels):
                        await enqueue_nowait(self._persistence_queue, PERSIST_DONE)

                    self._shutdown_tokens_seen += 1
                    outstanding_workers = max(
                        self._worker_count - self._shutdown_tokens_seen, 0
                    )
                    additional_tokens = max(
                        outstanding_workers - drained_sentinels, 0
                    )
                    if additional_tokens:
                        for _ in range(additional_tokens):
                            await enqueue_nowait(
                                self._persistence_queue, PERSIST_DONE
                            )
                    self._logger.debug(
                        "Persistence queue sentinel received; finishing run for worker %d.",
                        worker_id,
                    )
                    if drained_retry and not self._retry_queue.empty():
                        self._logger.warning(
                            "Retry queue still contains %d batch(es) after flush.",
                            self._retry_queue.qsize(),
                        )
                    return

                queue_size = self._persistence_queue.qsize()
                self._logger.info(
                    "Upsert worker %d handling %d points (queue size=%d)",
                    worker_id,
                    len(payload),
                    queue_size,
                )
                await self._upsert_fn(payload)
                if self._on_batch_complete is not None:
                    self._on_batch_complete(
                        worker_id, len(payload), self._persistence_queue.qsize()
                    )
            finally:
                self._persistence_queue.task_done()
                if payload not in (None, PERSIST_DONE):
                    self._upsert_semaphore.release()
