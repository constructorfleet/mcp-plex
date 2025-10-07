import asyncio
import logging
from unittest.mock import Mock, call, create_autospec

import pytest
from plexapi.server import PlexServer
from plexapi.video import Episode, Movie, Season, Show

from mcp_plex.common.types import AggregatedItem, PlexItem
from mcp_plex.loader.pipeline.channels import (
    INGEST_DONE,
    EpisodeBatch,
    IngestSentinel,
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
            plex_server=create_autospec(PlexServer, instance=True),
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
    async def scenario() -> tuple[SampleBatch | None, None | IngestSentinel, bool, int, int]:
        queue: asyncio.Queue = asyncio.Queue()
        stage = IngestionStage(
            plex_server=None,
            sample_items=[],
            movie_batch_size=1,
            episode_batch_size=1,
            sample_batch_size=2,
            output_queue=queue,
            completion_sentinel=INGEST_DONE,
        )

        await stage.run()

        first = await queue.get()
        second = await queue.get()
        return first, second, queue.empty(), stage.items_ingested, stage.batches_ingested

    first, second, empty, items_ingested, batches_ingested = asyncio.run(scenario())

    assert first is None
    assert second is INGEST_DONE
    assert empty is True
    assert items_ingested == 0
    assert batches_ingested == 0


def test_ingestion_stage_sample_partial_batches() -> None:
    async def scenario() -> tuple[
        list[SampleBatch],
        None | SampleBatch,
        None | IngestSentinel,
        int,
        int,
    ]:
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
            completion_sentinel=INGEST_DONE,
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
    assert second_token is INGEST_DONE
    assert items_ingested == 3
    assert batches_ingested == 2


@pytest.mark.parametrize(
    ("movie_batch_size", "episode_batch_size", "expected_name"),
    [
        (0, 1, "movie_batch_size"),
        (1, 0, "episode_batch_size"),
        (-3, 1, "movie_batch_size"),
        (1, -7, "episode_batch_size"),
    ],
)
def test_ingestion_stage_ingest_plex_requires_positive_batch_sizes(
    movie_batch_size: int, episode_batch_size: int, expected_name: str
) -> None:
    async def scenario() -> ValueError:
        queue: asyncio.Queue = asyncio.Queue()
        stage = IngestionStage(
            plex_server=create_autospec(PlexServer, instance=True),
            sample_items=None,
            movie_batch_size=1,
            episode_batch_size=1,
            sample_batch_size=1,
            output_queue=queue,
            completion_sentinel=INGEST_DONE,
        )

        plex_server = create_autospec(PlexServer, instance=True)

        with pytest.raises(ValueError) as excinfo:
            await stage._ingest_plex(
                plex_server=plex_server,
                movie_batch_size=movie_batch_size,
                episode_batch_size=episode_batch_size,
                output_queue=queue,
                logger=stage.logger,
            )

        return excinfo.value

    error = asyncio.run(scenario())
    assert str(error) == f"{expected_name} must be positive"


def test_ingestion_stage_backpressure_handling() -> None:
    async def scenario() -> tuple[
        list[SampleBatch],
        None | SampleBatch,
        None | IngestSentinel,
        int,
        int,
    ]:
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
            completion_sentinel=INGEST_DONE,
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
    assert second_token is INGEST_DONE
    assert items_ingested == 2
    assert batches_ingested == 2


def test_ingestion_stage_ingest_plex_batches_movies_and_episodes(caplog) -> None:
    caplog.set_level(logging.INFO)

    async def scenario() -> tuple[list[MovieBatch | EpisodeBatch], int, int, Mock]:
        queue: asyncio.Queue = asyncio.Queue()

        movie_section = Mock()
        movies = [
            create_autospec(Movie, instance=True, title="Movie 1"),
            create_autospec(Movie, instance=True, title="Movie 2"),
            create_autospec(Movie, instance=True, title="Movie 3"),
        ]
        movie_section.totalSize = len(movies)

        def movie_search(*, container_start=None, container_size=None, **_kwargs):
            start = container_start or 0
            size = container_size or len(movies)
            return movies[start : start + size]

        movie_section.search.side_effect = movie_search

        def _episodes(titles: list[str]) -> list[Episode]:
            return [create_autospec(Episode, instance=True, title=title) for title in titles]

        show_a_season_1 = create_autospec(Season, instance=True)
        show_a_s1_eps = _episodes(["S01E01", "S01E02"])

        def show_a_s1_side_effect(*, container_start=None, container_size=None, **_kwargs):
            start = container_start or 0
            size = container_size or len(show_a_s1_eps)
            return show_a_s1_eps[start : start + size]

        show_a_season_1.episodes.side_effect = show_a_s1_side_effect

        show_a_season_2 = create_autospec(Season, instance=True)
        show_a_s2_eps = _episodes(["S01E03"])

        def show_a_s2_side_effect(*, container_start=None, container_size=None, **_kwargs):
            start = container_start or 0
            size = container_size or len(show_a_s2_eps)
            return show_a_s2_eps[start : start + size]

        show_a_season_2.episodes.side_effect = show_a_s2_side_effect

        show_a = create_autospec(Show, instance=True, title="Show A")
        show_a.seasons.return_value = [show_a_season_1, show_a_season_2]

        show_b_season_1 = create_autospec(Season, instance=True)
        show_b_s1_eps = _episodes(["S01E01", "S01E02"])

        def show_b_s1_side_effect(*, container_start=None, container_size=None, **_kwargs):
            start = container_start or 0
            size = container_size or len(show_b_s1_eps)
            return show_b_s1_eps[start : start + size]

        show_b_season_1.episodes.side_effect = show_b_s1_side_effect

        show_b = create_autospec(Show, instance=True, title="Show B")
        show_b.seasons.return_value = [show_b_season_1]

        shows = [show_a, show_b]
        show_section = Mock()
        show_section.totalSize = len(shows)

        def show_search(*, container_start=None, container_size=None, **_kwargs):
            start = container_start or 0
            size = container_size or len(shows)
            return shows[start : start + size]

        show_section.search.side_effect = show_search

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
            completion_sentinel=INGEST_DONE,
        )

        await stage._ingest_plex(
            plex_server=plex,
            movie_batch_size=2,
            episode_batch_size=2,
            output_queue=queue,
            logger=stage.logger,
        )

        batches: list[MovieBatch | EpisodeBatch] = []
        while not queue.empty():
            batches.append(await queue.get())

        return (
            batches,
            stage.items_ingested,
            stage.batches_ingested,
            library,
            movie_section,
            show_section,
            show_a_season_1,
            show_a_season_2,
            show_b_season_1,
        )

    (
        batches,
        items_ingested,
        batches_ingested,
        library,
        movie_section,
        show_section,
        show_a_season_1,
        show_a_season_2,
        show_b_season_1,
    ) = asyncio.run(scenario())

    assert library.section.call_args_list == [
        (("Movies",),),
        (("TV Shows",),),
    ]
    assert movie_section.search.call_args_list == [
        call(container_start=0, container_size=2),
        call(container_start=2, container_size=2),
    ]
    assert show_section.search.call_args_list == [
        call(container_start=0, container_size=2),
        call(container_start=2, container_size=2),
    ]
    assert show_a_season_1.episodes.call_args_list == [
        call(container_start=0, container_size=2),
        call(container_start=2, container_size=2),
    ]
    assert show_a_season_2.episodes.call_args_list == [
        call(container_start=0, container_size=2),
    ]
    assert show_b_season_1.episodes.call_args_list == [
        call(container_start=0, container_size=2),
        call(container_start=2, container_size=2),
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


def test_ingestion_stage_ingest_plex_large_library_batches(caplog) -> None:
    caplog.set_level(logging.INFO)

    async def scenario() -> tuple[
        list[MovieBatch | EpisodeBatch],
        int,
        int,
        list,
        list,
        list[list],
    ]:
        queue: asyncio.Queue = asyncio.Queue()

        movie_batch_size = 50
        episode_batch_size = 25

        movie_section = Mock()
        movie_count = 105
        movies = [
            create_autospec(Movie, instance=True, title=f"Movie {index + 1}")
            for index in range(movie_count)
        ]
        movie_section.totalSize = movie_count

        def movie_search(*, container_start=None, container_size=None, **_kwargs):
            start = container_start or 0
            size = container_size or movie_count
            return movies[start : start + size]

        movie_section.search.side_effect = movie_search

        episode_counts = [30, 25, 10]
        shows: list[Show] = []
        season_mocks: list[Season] = []
        show_section = Mock()
        show_section.totalSize = len(episode_counts)

        for show_index, episode_count in enumerate(episode_counts, start=1):
            show = create_autospec(Show, instance=True, title=f"Show {show_index}")
            season = create_autospec(Season, instance=True)
            episodes = [
                create_autospec(
                    Episode,
                    instance=True,
                    title=f"S{show_index:02d}E{episode_index + 1:02d}",
                )
                for episode_index in range(episode_count)
            ]

            def _make_side_effect(eps: list[Episode]):
                def _side_effect(*, container_start=None, container_size=None, **_kwargs):
                    start = container_start or 0
                    size = container_size or len(eps)
                    return eps[start : start + size]

                return _side_effect

            season.episodes.side_effect = _make_side_effect(episodes)
            show.seasons.return_value = [season]
            shows.append(show)
            season_mocks.append(season)

        def show_search(*, container_start=None, container_size=None, **_kwargs):
            start = container_start or 0
            size = container_size or len(shows)
            return shows[start : start + size]

        show_section.search.side_effect = show_search

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
            movie_batch_size=movie_batch_size,
            episode_batch_size=episode_batch_size,
            sample_batch_size=10,
            output_queue=queue,
            completion_sentinel=INGEST_DONE,
        )

        await stage._ingest_plex(
            plex_server=plex,
            movie_batch_size=movie_batch_size,
            episode_batch_size=episode_batch_size,
            output_queue=queue,
            logger=stage.logger,
        )

        batches: list[MovieBatch | EpisodeBatch] = []
        while not queue.empty():
            batches.append(await queue.get())

        return (
            batches,
            stage.items_ingested,
            stage.batches_ingested,
            movie_section.search.call_args_list,
            show_section.search.call_args_list,
            [season.episodes.call_args_list for season in season_mocks],
        )

    (
        batches,
        items_ingested,
        batches_ingested,
        movie_search_calls,
        show_search_calls,
        season_episode_calls,
    ) = asyncio.run(scenario())

    movie_batches = [batch for batch in batches if isinstance(batch, MovieBatch)]
    episode_batches = [batch for batch in batches if isinstance(batch, EpisodeBatch)]

    assert [len(batch.movies) for batch in movie_batches] == [50, 50, 5]
    assert [len(batch.episodes) for batch in episode_batches] == [25, 5, 25, 10]
    assert items_ingested == 105 + 65
    assert batches_ingested == 7

    assert movie_search_calls == [
        call(container_start=0, container_size=50),
        call(container_start=50, container_size=50),
        call(container_start=100, container_size=50),
    ]
    assert show_search_calls == [
        call(container_start=0, container_size=25),
    ]
    assert season_episode_calls[0] == [
        call(container_start=0, container_size=25),
        call(container_start=25, container_size=25),
    ]
    assert season_episode_calls[1] == [
        call(container_start=0, container_size=25),
        call(container_start=25, container_size=25),
    ]
    assert season_episode_calls[2] == [
        call(container_start=0, container_size=25),
    ]

    assert "Discovered 105 Plex movie(s) for ingestion." in caplog.messages
    assert "Discovered 3 Plex show(s) for ingestion." in caplog.messages
