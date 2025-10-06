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
from .channels import (
    EpisodeBatch,
    IngestQueue,
    MovieBatch,
    SampleBatch,
    chunk_sequence,
)


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

        mode = "sample" if self._sample_items is not None else "plex"
        self._logger.info(
            "Starting ingestion stage (%s mode) with movie batch size=%d, episode batch size=%d, sample batch size=%d.",
            mode,
            self._movie_batch_size,
            self._episode_batch_size,
            self._sample_batch_size,
        )
        if self._sample_items is not None:
            await self._run_sample_ingestion(self._sample_items)
        else:
            await self._run_plex_ingestion()

        self._logger.debug("Publishing ingestion completion sentinels to downstream stages.")
        await self._output_queue.put(None)
        await self._output_queue.put(self._completion_sentinel)
        self._logger.info(
            "Ingestion stage finished after queuing %d batch(es) covering %d item(s).",
            self._batches_ingested,
            self._items_ingested,
        )

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
        start_batches = self._batches_ingested
        start_items = self._items_ingested
        self._logger.info(
            "Beginning sample ingestion for %d item(s) with batch size=%d.",
            item_count,
            self._sample_batch_size,
        )
        self._logger.info(
            "Sample ingestion has not been ported yet; %d items queued for later.",
            item_count,
        )
        await self._enqueue_sample_batches(items)
        self._logger.info(
            "Queued %d sample batch(es) covering %d item(s).",
            self._batches_ingested - start_batches,
            self._items_ingested - start_items,
        )
        await asyncio.sleep(0)

    async def _run_plex_ingestion(self) -> None:
        """Placeholder hook for Plex-backed ingestion."""

        if self._plex_server is None:
            self._logger.warning("Plex server unavailable; skipping ingestion.")
        else:
            self._logger.info(
                "Beginning Plex ingestion with movie batch size=%d and episode batch size=%d.",
                self._movie_batch_size,
                self._episode_batch_size,
            )
            await self._ingest_plex(
                plex_server=self._plex_server,
                movie_batch_size=self._movie_batch_size,
                episode_batch_size=self._episode_batch_size,
                output_queue=self._output_queue,
                logger=self._logger,
            )
            self._logger.info(
                "Completed Plex ingestion; emitted %d batch(es) covering %d item(s).",
                self._batches_ingested,
                self._items_ingested,
            )
        await asyncio.sleep(0)

    async def _ingest_plex(
        self,
        *,
        plex_server: object,
        movie_batch_size: int,
        episode_batch_size: int,
        output_queue: IngestQueue,
        logger: logging.Logger,
    ) -> None:
        """Retrieve Plex media and place batches onto *output_queue*."""

        movies_attr = getattr(plex_server, "movies", [])
        movies_source = movies_attr() if callable(movies_attr) else movies_attr
        movies = list(movies_source)
        logger.info(
            "Discovered %d Plex movie(s) for ingestion.",
            len(movies),
        )
        movie_batches = 0
        for batch_index, chunk in enumerate(
            chunk_sequence(movies, movie_batch_size), start=1
        ):
            batch_movies = list(chunk)
            if not batch_movies:
                continue

            batch = MovieBatch(movies=batch_movies)
            await output_queue.put(batch)
            self._items_ingested += len(batch_movies)
            self._batches_ingested += 1
            movie_batches += 1
            logger.info(
                "Queued Plex movie batch %d with %d movies (total items=%d).",
                batch_index,
                len(batch_movies),
                self._items_ingested,
            )

        shows_attr = getattr(plex_server, "shows", [])
        shows_source = shows_attr() if callable(shows_attr) else shows_attr
        shows = list(shows_source)
        logger.info(
            "Discovered %d Plex show(s) for ingestion.",
            len(shows),
        )
        episode_batches = 0
        episode_total = 0
        for show in shows:
            show_title = getattr(show, "title", str(show))
            episodes_attr = getattr(show, "episodes", [])
            episodes_source = (
                episodes_attr() if callable(episodes_attr) else episodes_attr
            )
            episodes = list(episodes_source)
            if not episodes:
                logger.debug("Show %s yielded no episodes for ingestion.", show_title)
            for batch_index, chunk in enumerate(
                chunk_sequence(episodes, episode_batch_size), start=1
            ):
                batch_episodes = list(chunk)
                if not batch_episodes:
                    continue

                batch = EpisodeBatch(show=show, episodes=batch_episodes)
                await output_queue.put(batch)
                self._items_ingested += len(batch_episodes)
                self._batches_ingested += 1
                episode_batches += 1
                episode_total += len(batch_episodes)
                logger.info(
                    "Queued Plex episode batch %d for %s with %d episodes (total items=%d).",
                    batch_index,
                    show_title,
                    len(batch_episodes),
                    self._items_ingested,
                )

        logger.debug(
            "Plex ingestion summary: %d movie batch(es), %d episode batch(es), %d episode(s).",
            movie_batches,
            episode_batches,
            episode_total,
        )

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
            self._logger.debug(
                "Queued sample batch with %d item(s) (total items=%d).",
                len(batch_items),
                self._items_ingested,
            )
