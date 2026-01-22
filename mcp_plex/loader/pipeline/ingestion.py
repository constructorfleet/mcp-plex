"""Ingestion stage coordinator for the loader pipeline.

At the moment the module only wires the configuration needed by the real
implementation.  The heavy lifting will be ported in subsequent commits, but
having the stage skeleton in place allows other components to depend on the
interface.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Iterable, Sequence

from ...common.types import AggregatedItem
from ...common.validation import require_positive
from .channels import (
    EpisodeBatch,
    IngestQueue,
    IngestSentinel,
    MovieBatch,
    SampleBatch,
    SeasonBatch,
    chunk_sequence,
    enqueue_nowait,
)

from plexapi.library import LibrarySection
from plexapi.server import PlexServer
from plexapi.video import Episode, Movie, Season, Show


class IngestionStage:
    """Coordinate ingesting items from Plex or bundled sample data."""

    def __init__(
        self,
        *,
        plex_server: PlexServer | None,
        sample_items: Sequence[AggregatedItem] | None,
        movie_batch_size: int,
        episode_batch_size: int,
        season_batch_size: int,
        show_batch_size: int,
        sample_batch_size: int,
        output_queue: IngestQueue,
        completion_sentinel: IngestSentinel,
    ) -> None:
        self._plex_server = plex_server
        self._sample_items = list(sample_items) if sample_items is not None else None
        self._movie_batch_size = int(movie_batch_size)
        self._episode_batch_size = int(episode_batch_size)
        self._season_batch_size = int(season_batch_size)
        self._show_batch_size = int(show_batch_size)
        self._sample_batch_size = int(sample_batch_size)
        self._output_queue = output_queue
        self._completion_sentinel: IngestSentinel = completion_sentinel
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

        self._logger.debug(
            "Publishing ingestion completion sentinels to downstream stages."
        )
        await enqueue_nowait(self._output_queue, None)
        await enqueue_nowait(self._output_queue, self._completion_sentinel)
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
                season_batch_size=self._season_batch_size,
                show_batch_size=self._show_batch_size,
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
        plex_server: PlexServer,
        movie_batch_size: int,
        episode_batch_size: int,
        season_batch_size: int,
        show_batch_size: int,
        output_queue: IngestQueue,
        logger: logging.Logger,
    ) -> None:
        """Retrieve Plex media and place batches onto *output_queue*."""

        movie_batch_size = require_positive(
            int(movie_batch_size),
            name="movie_batch_size",
        )
        episode_batch_size = require_positive(
            int(episode_batch_size),
            name="episode_batch_size",
        )
        season_batch_size = require_positive(
            int(season_batch_size),
            name="season_batch_size",
        )
        show_batch_size = require_positive(
            int(show_batch_size),
            name="show_batch_size",
        )

        library = plex_server.library

        def _log_discovered_count(
            *, section: LibrarySection, descriptor: str
        ) -> int | None:
            total = getattr(section, "totalSize", None)
            if isinstance(total, int):
                logger.info(
                    "Discovered %d Plex %s(s) for ingestion.",
                    total,
                    descriptor,
                )
                return total
            return None

        def _iter_section_items(
            *,
            fetch_page: Callable[[int], Sequence[Movie | Show]],
            batch_size: int,
        ) -> Iterable[Sequence[Movie | Show]]:
            start = 0
            while True:
                page = list(fetch_page(start))
                if not page:
                    break
                yield page
                if len(page) < batch_size:
                    break
                start += len(page)

        movies_section = library.section("Movies")
        discovered_movies = _log_discovered_count(
            section=movies_section,
            descriptor="movie",
        )

        movie_batches = 0
        movie_total = 0

        # Ensure only movies are processed in MovieBatch
        def _fetch_movies(start: int) -> Sequence[Movie]:
            return [item for item in movies_section.search(
                container_start=start,
                container_size=movie_batch_size,
            ) if isinstance(item, Movie)]

        for batch_index, batch_movies in enumerate(
            _iter_section_items(fetch_page=_fetch_movies, batch_size=movie_batch_size),
            start=1,
        ):
            if not batch_movies:
                continue

            batch = MovieBatch(movies=list(batch_movies))
            await enqueue_nowait(output_queue, batch)
            self._items_ingested += len(batch_movies)
            self._batches_ingested += 1
            movie_batches += 1
            movie_total += len(batch_movies)
            logger.info(
                "Queued Plex movie batch %d with %d movies (total items=%d).",
                batch_index,
                len(batch_movies),
                self._items_ingested,
            )

        if discovered_movies is None:
            logger.info(
                "Discovered %d Plex movie(s) for ingestion.",
                movie_total,
            )

        shows_section = library.section("TV Shows")
        discovered_shows = _log_discovered_count(
            section=shows_section,
            descriptor="show",
        )

        def _fetch_shows(start: int) -> Sequence[Show]:
            return shows_section.search(
                container_start=start,
                container_size=max(1, episode_batch_size),
            )

        show_total = 0
        episode_batches = 0
        episode_total = 0

        for show_batch in _iter_section_items(
            fetch_page=_fetch_shows,
            batch_size=max(1, episode_batch_size),
        ):
            for show in show_batch:
                show_total += 1
                show_title = getattr(show, "title", str(show))
                show_episode_count = 0

                # Initialize variables to avoid unbound errors
                pending_episodes: list[Episode] = []
                show_batch_index = 0

                seasons: Sequence[Season] = show.seasons()
                for season in seasons:
                    start = 0
                    while True:
                        season_page = list(
                            season.episodes(
                                container_start=start,
                                container_size=episode_batch_size,
                            )
                        )
                        if not season_page:
                            break

                        show_episode_count += len(season_page)
                        pending_episodes.extend(season_page)

                        while len(pending_episodes) >= episode_batch_size:
                            batch_episodes = pending_episodes[:episode_batch_size]
                            pending_episodes = pending_episodes[episode_batch_size:]
                            show_batch_index += 1
                            batch = EpisodeBatch(
                                show=show,
                                episodes=list(batch_episodes),
                            )
                            await enqueue_nowait(output_queue, batch)
                            self._items_ingested += len(batch_episodes)
                            self._batches_ingested += 1
                            episode_batches += 1
                            episode_total += len(batch_episodes)
                            logger.info(
                                "Queued Plex episode batch %d for %s with %d episodes (total items=%d).",
                                show_batch_index,
                                show_title,
                                len(batch_episodes),
                                self._items_ingested,
                            )

                        if len(season_page) < episode_batch_size:
                            break
                        start += len(season_page)

                if pending_episodes:
                    show_batch_index += 1
                    batch = EpisodeBatch(
                        show=show,
                        episodes=list(pending_episodes),
                    )
                    await enqueue_nowait(output_queue, batch)
                    self._items_ingested += len(pending_episodes)
                    self._batches_ingested += 1
                    episode_batches += 1
                    episode_total += len(pending_episodes)
                    logger.info(
                        "Queued Plex episode batch %d for %s with %d episodes (total items=%d).",
                        show_batch_index,
                        show_title,
                        len(pending_episodes),
                        self._items_ingested,
                    )

                if show_episode_count == 0:
                    logger.debug(
                        "Show %s yielded no episodes for ingestion.",
                        show_title,
                    )

        if discovered_shows is None:
            logger.info(
                "Discovered %d Plex show(s) for ingestion.",
                show_total,
            )

        # Process seasons
        season_batches = 0
        season_total = 0

        # Refine seasons handling to ensure correct types
        shows_for_seasons = shows_section.all()
        if not isinstance(shows_for_seasons, Iterable):
            shows_for_seasons = []
        for show in shows_for_seasons:
            if not isinstance(show, Show):
                continue

            show_title = getattr(show, "title", str(show))
            pending_seasons: list[Season] = []

            seasons = [season for season in (show.seasons() or []) if isinstance(season, Season)]

            # Ensure only valid episodes are processed and track seasons for batching
            for season in seasons:
                pending_seasons.append(season)

                while len(pending_seasons) >= season_batch_size:
                    batch_seasons = pending_seasons[:season_batch_size]
                    pending_seasons = pending_seasons[season_batch_size:]
                    batch = SeasonBatch(
                        show=show,
                        seasons=list(batch_seasons),
                    )
                    await enqueue_nowait(output_queue, batch)
                    self._items_ingested += len(batch_seasons)
                    self._batches_ingested += 1
                    season_batches += 1
                    season_total += len(batch_seasons)
                    logger.info(
                        "Queued Plex season batch for %s with %d seasons (total items=%d).",
                        show_title,
                        len(batch_seasons),
                        self._items_ingested,
                    )

            if pending_seasons:
                batch = SeasonBatch(
                    show=show,
                    seasons=list(pending_seasons),
                )
                await enqueue_nowait(output_queue, batch)
                self._items_ingested += len(pending_seasons)
                self._batches_ingested += 1
                season_batches += 1
                season_total += len(pending_seasons)
                logger.info(
                    "Queued Plex season batch for %s with %d seasons (total items=%d).",
                    show_title,
                    len(pending_seasons),
                    self._items_ingested,
                )

        logger.debug(
            "Plex ingestion summary: %d movie batch(es), %d episode batch(es), %d season batch(es), %d episode(s), %d season(s).",
            movie_batches,
            episode_batches,
            season_batches,
            episode_total,
            season_total,
        )

    async def _enqueue_sample_batches(self, items: Sequence[AggregatedItem]) -> None:
        """Place sample items onto the ingest queue in configured batch sizes."""

        for chunk in chunk_sequence(items, self._sample_batch_size):
            batch_items = list(chunk)
            if not batch_items:
                continue

            await enqueue_nowait(self._output_queue, SampleBatch(items=batch_items))
            self._items_ingested += len(batch_items)
            self._batches_ingested += 1
            self._logger.debug(
                "Queued sample batch with %d item(s) (total items=%d).",
                len(batch_items),
                self._items_ingested,
            )
