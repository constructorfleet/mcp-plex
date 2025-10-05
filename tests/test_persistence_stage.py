import asyncio

from mcp_plex.loader.pipeline.channels import PersistenceQueue
from mcp_plex.loader.pipeline.persistence import PersistenceStage


class _FakeQdrantClient:
    pass


def test_persistence_stage_logger_name() -> None:
    async def scenario() -> str:
        client = _FakeQdrantClient()
        persistence_queue: PersistenceQueue = asyncio.Queue()
        retry_queue: asyncio.Queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(3)

        stage = PersistenceStage(
            client=client,
            collection_name="media-items",
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            persistence_queue=persistence_queue,
            retry_queue=retry_queue,
            upsert_semaphore=semaphore,
        )
        return stage.logger.name

    logger_name = asyncio.run(scenario())

    assert logger_name == "mcp_plex.loader.persistence"


def test_persistence_stage_holds_dependencies() -> None:
    async def scenario() -> tuple[PersistenceStage, _FakeQdrantClient, PersistenceQueue, asyncio.Queue, asyncio.Semaphore]:
        client = _FakeQdrantClient()
        persistence_queue: PersistenceQueue = asyncio.Queue()
        retry_queue: asyncio.Queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(5)

        stage = PersistenceStage(
            client=client,
            collection_name="media-items",
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            persistence_queue=persistence_queue,
            retry_queue=retry_queue,
            upsert_semaphore=semaphore,
        )

        return stage, client, persistence_queue, retry_queue, semaphore

    stage, client, persistence_queue, retry_queue, semaphore = asyncio.run(scenario())

    assert stage.qdrant_client is client
    assert stage.collection_name == "media-items"
    assert stage.dense_vector_name == "dense"
    assert stage.sparse_vector_name == "sparse"
    assert stage.persistence_queue is persistence_queue
    assert stage.retry_queue is retry_queue
    assert stage.upsert_semaphore is semaphore
