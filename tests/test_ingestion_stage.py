import asyncio

from mcp_plex.common.types import AggregatedItem, PlexItem
from mcp_plex.loader.pipeline.channels import INGEST_DONE, SampleBatch
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
