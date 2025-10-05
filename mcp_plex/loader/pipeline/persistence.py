"""Persistence stage placeholder used by the loader pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Sequence

from .channels import PersistenceQueue, chunk_sequence, require_positive

if TYPE_CHECKING:  # pragma: no cover - typing helpers only
    from qdrant_client import AsyncQdrantClient, models

    PersistencePayload = list[models.PointStruct]
else:  # pragma: no cover - runtime fallback when qdrant_client is absent
    AsyncQdrantClient = Any  # type: ignore[assignment]
    PersistencePayload = list[Any]


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

    async def enqueue_points(
        self, points: Sequence["models.PointStruct"]
    ) -> None:
        """Chunk *points* and place them on the persistence queue."""

        if not points:
            return

        for chunk in chunk_sequence(list(points), self._upsert_buffer_size):
            batch = list(chunk)
            if not batch:
                continue
            await self._upsert_semaphore.acquire()
            try:
                await self._persistence_queue.put(batch)
            except BaseException:
                self._upsert_semaphore.release()
                raise

    async def run(self, worker_id: int) -> None:
        """Drain the persistence queue until a sentinel is received."""

        while True:
            payload = await self._persistence_queue.get()
            try:
                if payload is None:
                    self._logger.debug(
                        "Persistence queue sentinel received; finishing run for worker %d.",
                        worker_id,
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
                if payload is not None:
                    self._upsert_semaphore.release()
