import asyncio
import logging

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
            plex_server=object(),
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

    class FakeMovie:
        def __init__(self, title: str) -> None:
            self.title = title

    class FakeEpisode:
        def __init__(self, title: str) -> None:
            self.title = title

    class FakeShow:
        def __init__(self, title: str, episode_titles: list[str]) -> None:
            self.title = title
            self._episodes = [FakeEpisode(ep_title) for ep_title in episode_titles]

        def episodes(self) -> list[FakeEpisode]:
            return list(self._episodes)

    class FakePlex:
        def __init__(self) -> None:
            self._movies = [
                FakeMovie("Movie 1"),
                FakeMovie("Movie 2"),
                FakeMovie("Movie 3"),
            ]
            self._shows = [
                FakeShow("Show A", ["S01E01", "S01E02", "S01E03"]),
                FakeShow("Show B", ["S01E01", "S01E02"]),
            ]

        def movies(self) -> list[FakeMovie]:
            return list(self._movies)

        def shows(self) -> list[FakeShow]:
            return list(self._shows)

    sentinel = object()

    async def scenario() -> tuple[list[object], int, int]:
        queue: asyncio.Queue = asyncio.Queue()
        plex = FakePlex()
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

        return batches, stage.items_ingested, stage.batches_ingested

    batches, items_ingested, batches_ingested = asyncio.run(scenario())

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

    assert caplog.messages == [
        "Queued Plex movie batch 1 with 2 movies (total items=2).",
        "Queued Plex movie batch 2 with 1 movies (total items=3).",
        "Queued Plex episode batch 1 for Show A with 2 episodes (total items=5).",
        "Queued Plex episode batch 2 for Show A with 1 episodes (total items=6).",
        "Queued Plex episode batch 1 for Show B with 2 episodes (total items=8).",
    ]
