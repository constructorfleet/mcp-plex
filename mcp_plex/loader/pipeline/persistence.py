"""Persistence stage placeholder used by the loader pipeline."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .channels import PersistenceQueue

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
    ) -> None:
        self._client = client
        self._collection_name = str(collection_name)
        self._dense_vector_name = str(dense_vector_name)
        self._sparse_vector_name = str(sparse_vector_name)
        self._persistence_queue = persistence_queue
        self._retry_queue = retry_queue
        self._upsert_semaphore = upsert_semaphore
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

    async def run(self) -> None:
        """Drain the persistence queue until a sentinel is received."""

        while True:
            payload = await self._persistence_queue.get()
            try:
                if payload is None:
                    self._logger.debug(
                        "Persistence queue sentinel received; finishing placeholder run."
                    )
                    return

                self._logger.debug(
                    "Placeholder persistence stage received batch with %d items.",
                    len(payload),
                )
            finally:
                self._persistence_queue.task_done()

            await asyncio.sleep(0)
