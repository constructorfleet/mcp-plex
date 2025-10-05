import asyncio

from mcp_plex.common.types import AggregatedItem, PlexItem
from mcp_plex.loader.pipeline.channels import INGEST_DONE
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


def test_ingestion_stage_emits_completion_sentinels() -> None:
    sentinel = object()

    async def scenario() -> tuple[object | None, object | None, bool]:
        queue: asyncio.Queue = asyncio.Queue()
        sample_items = [_make_aggregated_item("1"), _make_aggregated_item("2")]
        stage = IngestionStage(
            plex_server=None,
            sample_items=sample_items,
            movie_batch_size=1,
            episode_batch_size=1,
            sample_batch_size=1,
            output_queue=queue,
            completion_sentinel=sentinel,
        )

        await stage.run()

        first = await queue.get()
        second = await queue.get()
        return first, second, queue.empty()

    first, second, empty = asyncio.run(scenario())

    assert first is None
    assert second is sentinel
    assert empty is True
