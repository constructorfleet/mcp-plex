"""Enrichment stage coordinator for the loader pipeline.

The real enrichment logic pulls additional metadata from TMDb and IMDb before
handing the enriched payloads off to the persistence stage.  Only the stage
scaffolding is implemented for now so other pipeline components can start
interacting with a consistent interface while the remaining logic is ported in
follow-up changes.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from .channels import (
    EpisodeBatch,
    IMDbRetryQueue,
    INGEST_DONE,
    IngestQueue,
    MovieBatch,
    PersistenceQueue,
    SampleBatch,
)


class EnrichmentStage:
    """Coordinate metadata enrichment for ingested media batches."""

    def __init__(
        self,
        *,
        http_client_factory: Callable[[], Awaitable[Any] | Any],
        tmdb_api_key: str,
        ingest_queue: IngestQueue,
        persistence_queue: PersistenceQueue,
        imdb_retry_queue: IMDbRetryQueue | None,
        movie_batch_size: int,
        episode_batch_size: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self._http_client_factory = http_client_factory
        self._tmdb_api_key = str(tmdb_api_key)
        self._ingest_queue = ingest_queue
        self._persistence_queue = persistence_queue
        self._imdb_retry_queue = imdb_retry_queue or IMDbRetryQueue()
        self._movie_batch_size = int(movie_batch_size)
        self._episode_batch_size = int(episode_batch_size)
        self._logger = logger or logging.getLogger("mcp_plex.loader.enrichment")

    @property
    def logger(self) -> logging.Logger:
        """Logger used by the enrichment stage."""

        return self._logger

    @property
    def imdb_retry_queue(self) -> IMDbRetryQueue:
        """IMDb retry queue used by the enrichment stage."""

        return self._imdb_retry_queue

    async def run(self) -> None:
        """Execute the enrichment stage."""

        while True:
            batch = await self._ingest_queue.get()

            if batch is None:
                self._logger.debug("Received legacy completion token; ignoring.")
                continue

            if batch is INGEST_DONE:
                self._logger.info("Ingestion completed; finishing enrichment stage.")
                break

            if isinstance(batch, MovieBatch):
                await self._handle_movie_batch(batch)
            elif isinstance(batch, EpisodeBatch):
                await self._handle_episode_batch(batch)
            elif isinstance(batch, SampleBatch):
                await self._handle_sample_batch(batch)
            else:  # pragma: no cover - defensive logging for future types
                self._logger.warning("Received unsupported batch type: %r", batch)

        await self._persistence_queue.put(None)

    async def _handle_movie_batch(self, batch: MovieBatch) -> None:
        """Placeholder hook for processing Plex movie batches."""

        movie_count = len(batch.movies)
        self._logger.info(
            "Movie enrichment has not been ported yet; %d movies queued for later.",
            movie_count,
        )
        await asyncio.sleep(0)

    async def _handle_episode_batch(self, batch: EpisodeBatch) -> None:
        """Placeholder hook for processing Plex episode batches."""

        show_title = getattr(batch.show, "title", str(batch.show))
        episode_count = len(batch.episodes)
        self._logger.info(
            "Episode enrichment has not been ported yet; %d episodes pending for %s.",
            episode_count,
            show_title,
        )
        await asyncio.sleep(0)

    async def _handle_sample_batch(self, batch: SampleBatch) -> None:
        """Placeholder hook for processing sample data batches."""

        item_count = len(batch.items)
        self._logger.info(
            "Sample enrichment has not been ported yet; %d items queued for later.",
            item_count,
        )
        await asyncio.sleep(0)
