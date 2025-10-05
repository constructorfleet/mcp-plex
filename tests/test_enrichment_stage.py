import asyncio
import logging

from mcp_plex.common.types import AggregatedItem, PlexItem
from mcp_plex.loader.pipeline.channels import (
    EpisodeBatch,
    IMDbRetryQueue,
    INGEST_DONE,
    MovieBatch,
    SampleBatch,
)
from mcp_plex.loader.pipeline.enrichment import EnrichmentStage


def _make_aggregated_item(rating_key: str) -> AggregatedItem:
    return AggregatedItem(
        plex=PlexItem(
            rating_key=rating_key,
            guid=f"plex://{rating_key}",
            type="movie",
            title=f"Title {rating_key}",
        )
    )


def test_enrichment_stage_logger_name() -> None:
    async def scenario() -> str:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="tmdb",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=25,
            episode_batch_size=10,
        )
        return stage.logger.name

    logger_name = asyncio.run(scenario())
    assert logger_name == "mcp_plex.loader.enrichment"


def test_enrichment_stage_handles_batches_and_completion(caplog) -> None:
    caplog.set_level(logging.INFO)

    class FakeShow:
        def __init__(self, title: str) -> None:
            self.title = title

    async def scenario() -> tuple[list[str], object | None]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="tmdb",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=50,
            episode_batch_size=25,
        )

        await ingest_queue.put(MovieBatch(movies=[object(), object()]))
        await ingest_queue.put(
            EpisodeBatch(show=FakeShow("Example Show"), episodes=[object()])
        )
        await ingest_queue.put(SampleBatch(items=[_make_aggregated_item("1")]))
        await ingest_queue.put(None)
        await ingest_queue.put(INGEST_DONE)

        await stage.run()

        completion_token = await persistence_queue.get()
        messages = [record.getMessage() for record in caplog.records]
        return messages, completion_token

    messages, completion_token = asyncio.run(scenario())

    assert any("Movie enrichment has not been ported yet" in message for message in messages)
    assert any("Episode enrichment has not been ported yet" in message for message in messages)
    assert any("Sample enrichment has not been ported yet" in message for message in messages)
    assert completion_token is None


def test_enrichment_stage_uses_injected_imdb_retry_queue() -> None:
    async def scenario() -> bool:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        retry_queue = IMDbRetryQueue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="tmdb",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=retry_queue,
            movie_batch_size=25,
            episode_batch_size=10,
        )
        return stage.imdb_retry_queue is retry_queue

    same_queue = asyncio.run(scenario())
    assert same_queue is True


def test_enrichment_stage_creates_retry_queue_when_missing() -> None:
    async def scenario() -> IMDbRetryQueue:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="tmdb",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=None,
            movie_batch_size=25,
            episode_batch_size=10,
        )
        return stage.imdb_retry_queue

    retry_queue = asyncio.run(scenario())
    assert isinstance(retry_queue, IMDbRetryQueue)
