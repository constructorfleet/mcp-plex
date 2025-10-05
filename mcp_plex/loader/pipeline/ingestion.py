"""Ingestion stage coordinator for the loader pipeline.

At the moment the module only wires the configuration needed by the real
implementation.  The heavy lifting will be ported in subsequent commits, but
having the stage skeleton in place allows other components to depend on the
interface.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Sequence

from ...common.types import AggregatedItem
from .channels import IngestQueue, SampleBatch, chunk_sequence


class IngestionStage:
    """Coordinate ingesting items from Plex or bundled sample data."""

    def __init__(
        self,
        *,
        plex_server: object | None,
        sample_items: Sequence[AggregatedItem] | None,
        movie_batch_size: int,
        episode_batch_size: int,
        sample_batch_size: int,
        output_queue: IngestQueue,
        completion_sentinel: object,
    ) -> None:
        self._plex_server = plex_server
        self._sample_items = list(sample_items) if sample_items is not None else None
        self._movie_batch_size = int(movie_batch_size)
        self._episode_batch_size = int(episode_batch_size)
        self._sample_batch_size = int(sample_batch_size)
        self._output_queue = output_queue
        self._completion_sentinel = completion_sentinel
        self._logger = logging.getLogger("mcp_plex.loader.ingestion")
        self._items_ingested = 0
        self._batches_ingested = 0

    @property
    def logger(self) -> logging.Logger:
        """Logger used by the ingestion stage."""

        return self._logger

    async def run(self) -> None:
        """Execute the ingestion stage.

        Sample data takes precedence over Plex driven ingestion.  The
        placeholders invoked here will be replaced as richer implementations are
        ported from the legacy loader.
        """

        if self._sample_items is not None:
            await self._run_sample_ingestion(self._sample_items)
        else:
            await self._run_plex_ingestion()

        await self._output_queue.put(None)
        await self._output_queue.put(self._completion_sentinel)

    @property
    def items_ingested(self) -> int:
        """Total number of items placed onto the ingest queue."""

        return self._items_ingested

    @property
    def batches_ingested(self) -> int:
        """Total number of batches placed onto the ingest queue."""

        return self._batches_ingested

    async def _run_sample_ingestion(self, items: Sequence[AggregatedItem]) -> None:
        """Placeholder hook for the sample ingestion flow."""

        item_count = len(items)
        self._logger.info(
            "Sample ingestion has not been ported yet; %d items queued for later.",
            item_count,
        )
        await self._enqueue_sample_batches(items)
        await asyncio.sleep(0)

    async def _run_plex_ingestion(self) -> None:
        """Placeholder hook for Plex-backed ingestion."""

        if self._plex_server is None:
            self._logger.warning("Plex server unavailable; skipping ingestion.")
        else:
            self._logger.info("Plex ingestion pending implementation.")
        await asyncio.sleep(0)

    async def _enqueue_sample_batches(
        self, items: Sequence[AggregatedItem]
    ) -> None:
        """Place sample items onto the ingest queue in configured batch sizes."""

        for chunk in chunk_sequence(items, self._sample_batch_size):
            batch_items = list(chunk)
            if not batch_items:
                continue

            await self._output_queue.put(SampleBatch(items=batch_items))
            self._items_ingested += len(batch_items)
            self._batches_ingested += 1
