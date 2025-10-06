import asyncio
import logging
from typing import cast
from unittest.mock import Mock, create_autospec

import pytest
from plexapi.server import PlexServer
from plexapi.video import Episode, Movie, Season, Show

from mcp_plex.common.types import AggregatedItem, PlexItem
from mcp_plex.loader.pipeline.channels import (
    INGEST_DONE,
    EpisodeBatch,
    MovieBatch,
    SampleBatch,
)
from mcp_plex.loader.pipeline.ingestion import IngestionStage


def _make_aggregated_item(rating_key: str) -> AggregatedItem:
    return AggregatedItem(
        plex=PlexItem(
            rating_key=rating_key,
            guid=f"plex://{rating_key}",
            type="movie",
            title=f"Title {rating_key}",
        )
    )


def test_ingestion_stage_logger_name() -> None:
    async def scenario() -> str:
        queue: asyncio.Queue = asyncio.Queue()
        stage = IngestionStage(
            plex_server=cast(PlexServer, object()),
            sample_items=None,
            movie_batch_size=50,
            episode_batch_size=25,
            sample_batch_size=10,
            output_queue=queue,
            completion_sentinel=INGEST_DONE,
        )
        return stage.logger.name

    logger_name = asyncio.run(scenario())
    assert logger_name == "mcp_plex.loader.ingestion"


def test_ingestion_stage_sample_empty_batches() -> None:
    sentinel = object()

    async def scenario() -> tuple[object | None, object | None, bool, int, int]:
        queue: asyncio.Queue = asyncio.Queue()
        stage = IngestionStage(
            plex_server=None,
            sample_items=[],
            movie_batch_size=1,
            episode_batch_size=1,
            sample_batch_size=2,
            output_queue=queue,
            completion_sentinel=sentinel,
        )

        await stage.run()

        first = await queue.get()
        second = await queue.get()
        return first, second, queue.empty(), stage.items_ingested, stage.batches_ingested

    first, second, empty, items_ingested, batches_ingested = asyncio.run(scenario())

    assert first is None
    assert second is sentinel
    assert empty is True
    assert items_ingested == 0
    assert batches_ingested == 0


def test_ingestion_stage_sample_partial_batches() -> None:
    sentinel = object()

    async def scenario() -> tuple[list[SampleBatch], object | None, object | None, int, int]:
        queue: asyncio.Queue = asyncio.Queue()
        sample_items = [
            _make_aggregated_item("1"),
            _make_aggregated_item("2"),
            _make_aggregated_item("3"),
        ]
        stage = IngestionStage(
            plex_server=None,
            sample_items=sample_items,
            movie_batch_size=1,
            episode_batch_size=1,
            sample_batch_size=2,
            output_queue=queue,
            completion_sentinel=sentinel,
        )

        await stage.run()

        batches = [await queue.get(), await queue.get()]
        first_token = await queue.get()
        second_token = await queue.get()
        return batches, first_token, second_token, stage.items_ingested, stage.batches_ingested

    batches, first_token, second_token, items_ingested, batches_ingested = asyncio.run(
        scenario()
    )

    assert all(isinstance(batch, SampleBatch) for batch in batches)
    assert [len(batch.items) for batch in batches] == [2, 1]
    assert first_token is None
    assert second_token is sentinel
    assert items_ingested == 3
    assert batches_ingested == 2


def test_ingestion_stage_backpressure_handling() -> None:
    sentinel = object()

    async def scenario() -> tuple[list[SampleBatch], object | None, object | None, int, int]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        sample_items = [
            _make_aggregated_item("1"),
            _make_aggregated_item("2"),
        ]
        stage = IngestionStage(
            plex_server=None,
            sample_items=sample_items,
            movie_batch_size=1,
            episode_batch_size=1,
            sample_batch_size=1,
            output_queue=queue,
            completion_sentinel=sentinel,
        )

        run_task = asyncio.create_task(stage.run())
        await asyncio.sleep(0)
        assert run_task.done() is False

        first_batch = await queue.get()
        assert isinstance(first_batch, SampleBatch)
        await asyncio.sleep(0)

        second_batch = await queue.get()
        first_token = await queue.get()
        second_token = await queue.get()

        await run_task
        return [first_batch, second_batch], first_token, second_token, stage.items_ingested, stage.batches_ingested

    batches, first_token, second_token, items_ingested, batches_ingested = asyncio.run(
        scenario()
    )

    assert [len(batch.items) for batch in batches] == [1, 1]
    assert first_token is None
    assert second_token is sentinel
    assert items_ingested == 2
    assert batches_ingested == 2


def test_ingestion_stage_ingest_plex_batches_movies_and_episodes(caplog) -> None:
    caplog.set_level(logging.INFO)

    sentinel = object()

    async def scenario() -> tuple[list[object], int, int, Mock]:
        queue: asyncio.Queue = asyncio.Queue()

        movie_section = Mock()
        movies = [
            create_autospec(Movie, instance=True, title="Movie 1"),
            create_autospec(Movie, instance=True, title="Movie 2"),
            create_autospec(Movie, instance=True, title="Movie 3"),
        ]
        movie_section.all.return_value = movies

        def _episodes(titles: list[str]) -> list[Episode]:
            return [create_autospec(Episode, instance=True, title=title) for title in titles]

        show_a_season_1 = create_autospec(Season, instance=True)
        show_a_season_1.episodes.return_value = _episodes(["S01E01", "S01E02"])
        show_a_season_2 = create_autospec(Season, instance=True)
        show_a_season_2.episodes.return_value = _episodes(["S01E03"])

        show_a = create_autospec(Show, instance=True, title="Show A")
        show_a.seasons.return_value = [show_a_season_1, show_a_season_2]

        show_b_season_1 = create_autospec(Season, instance=True)
        show_b_season_1.episodes.return_value = _episodes(["S01E01", "S01E02"])

        show_b = create_autospec(Show, instance=True, title="Show B")
        show_b.seasons.return_value = [show_b_season_1]

        shows = [show_a, show_b]
        show_section = Mock()
        show_section.all.return_value = shows

        library = Mock()
        library.section.side_effect = lambda name: {
            "Movies": movie_section,
            "TV Shows": show_section,
        }[name]

        plex = create_autospec(PlexServer, instance=True)
        plex.library = library

        stage = IngestionStage(
            plex_server=plex,
            sample_items=None,
            movie_batch_size=2,
            episode_batch_size=2,
            sample_batch_size=10,
            output_queue=queue,
            completion_sentinel=sentinel,
        )

        await stage._ingest_plex(
            plex_server=plex,
            movie_batch_size=2,
            episode_batch_size=2,
            output_queue=queue,
            logger=stage.logger,
        )

        batches: list[object] = []
        while not queue.empty():
            batches.append(await queue.get())

        return batches, stage.items_ingested, stage.batches_ingested, library

    batches, items_ingested, batches_ingested, library = asyncio.run(scenario())

    assert library.section.call_args_list == [
        (("Movies",),),
        (("TV Shows",),),
    ]
    assert items_ingested == 8
    assert batches_ingested == 5

    assert len(batches) == 5
    assert isinstance(batches[0], MovieBatch)
    assert [movie.title for movie in batches[0].movies] == ["Movie 1", "Movie 2"]
    assert isinstance(batches[1], MovieBatch)
    assert [movie.title for movie in batches[1].movies] == ["Movie 3"]
    assert isinstance(batches[2], EpisodeBatch)
    assert [episode.title for episode in batches[2].episodes] == ["S01E01", "S01E02"]
    assert isinstance(batches[3], EpisodeBatch)
    assert [episode.title for episode in batches[3].episodes] == ["S01E03"]
    assert isinstance(batches[4], EpisodeBatch)
    assert [episode.title for episode in batches[4].episodes] == ["S01E01", "S01E02"]

    expected_tail = [
        "Queued Plex movie batch 1 with 2 movies (total items=2).",
        "Queued Plex movie batch 2 with 1 movies (total items=3).",
        "Queued Plex episode batch 1 for Show A with 2 episodes (total items=5).",
        "Queued Plex episode batch 2 for Show A with 1 episodes (total items=6).",
        "Queued Plex episode batch 1 for Show B with 2 episodes (total items=8).",
    ]
    observed_iter = iter(caplog.messages)
    for expected in expected_tail:
        for message in observed_iter:
            if message == expected:
                break
        else:  # pragma: no cover - pytest fail helper
            pytest.fail(f"Expected log message not found: {expected}")
    assert "Discovered 3 Plex movie(s) for ingestion." in caplog.messages
    assert "Discovered 2 Plex show(s) for ingestion." in caplog.messages
