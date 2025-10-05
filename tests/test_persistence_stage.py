import asyncio
from typing import Any

import pytest

from mcp_plex.loader import _upsert_in_batches
from mcp_plex.loader.pipeline.channels import PERSIST_DONE, PersistenceQueue
from mcp_plex.loader.pipeline.persistence import PersistenceStage


class _FakeQdrantClient:
    async def upsert(self, *, collection_name: str, points: list[Any]) -> None:
        raise NotImplementedError


async def _noop_upsert(_: list[Any]) -> None:
    await asyncio.sleep(0)


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
            upsert_buffer_size=2,
            upsert_fn=_noop_upsert,
            on_batch_complete=None,
        )
        return stage.logger.name

    logger_name = asyncio.run(scenario())

    assert logger_name == "mcp_plex.loader.persistence"


def test_persistence_stage_holds_dependencies() -> None:
    async def scenario() -> tuple[
        PersistenceStage,
        _FakeQdrantClient,
        PersistenceQueue,
        asyncio.Queue,
        asyncio.Semaphore,
    ]:
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
            upsert_buffer_size=3,
            upsert_fn=_noop_upsert,
            on_batch_complete=None,
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


def test_persistence_stage_upserts_batches() -> None:
    async def scenario() -> tuple[list[list[int]], list[tuple[int, int, int]], int]:
        persistence_queue: PersistenceQueue = asyncio.Queue()
        retry_queue: asyncio.Queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(2)

        processed: list[list[int]] = []

        async def fake_upsert(batch: list[int]) -> None:
            processed.append(list(batch))

        completions: list[tuple[int, int, int]] = []

        def on_batch_complete(worker_id: int, batch_size: int, queue_size: int) -> None:
            completions.append((worker_id, batch_size, queue_size))

        stage = PersistenceStage(
            client=_FakeQdrantClient(),
            collection_name="media-items",
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            persistence_queue=persistence_queue,
            retry_queue=retry_queue,
            upsert_semaphore=semaphore,
            upsert_buffer_size=2,
            upsert_fn=fake_upsert,
            on_batch_complete=on_batch_complete,
        )

        workers = [asyncio.create_task(stage.run(worker_id)) for worker_id in range(2)]

        await stage.enqueue_points([1, 2, 3])
        await persistence_queue.join()

        for _ in workers:
            await persistence_queue.put(PERSIST_DONE)

        await asyncio.gather(*workers)

        return processed, completions, semaphore._value  # type: ignore[attr-defined]

    processed, completions, semaphore_value = asyncio.run(scenario())

    assert {tuple(batch) for batch in processed} == {(1, 2), (3,)}
    assert sorted(batch_size for _, batch_size, _ in completions) == [1, 2]
    assert completions[-1][2] == 0
    assert semaphore_value == 2


def test_persistence_stage_populates_retry_queue_on_failure() -> None:
    async def scenario() -> list[list[int]]:
        persistence_queue: PersistenceQueue = asyncio.Queue()
        retry_queue: asyncio.Queue[list[list[int]]] = asyncio.Queue()
        semaphore = asyncio.Semaphore(1)

        class _FailingClient:
            async def upsert(
                self, *, collection_name: str, points: list[list[int]]
            ) -> None:
                raise RuntimeError("boom")

        async def upsert_fn(batch: list[list[int]]) -> None:
            await _upsert_in_batches(
                _FailingClient(),
                "media-items",
                batch,
                retry_queue=retry_queue,
            )

        stage = PersistenceStage(
            client=_FakeQdrantClient(),
            collection_name="media-items",
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            persistence_queue=persistence_queue,
            retry_queue=retry_queue,
            upsert_semaphore=semaphore,
            upsert_buffer_size=10,
            upsert_fn=upsert_fn,
            on_batch_complete=None,
        )

        worker = asyncio.create_task(stage.run(0))

        await stage.enqueue_points([[1, 2]])
        await persistence_queue.join()
        await persistence_queue.put(PERSIST_DONE)
        await asyncio.wait_for(asyncio.gather(worker), timeout=1)

        failures: list[list[int]] = []
        while not retry_queue.empty():
            failures.append(await retry_queue.get())

        return failures

    failures = asyncio.run(scenario())

    assert failures == [[[1, 2]]]


def test_persistence_stage_releases_semaphore_on_upsert_error() -> None:
    async def scenario() -> int:
        persistence_queue: PersistenceQueue = asyncio.Queue()
        retry_queue: asyncio.Queue = asyncio.Queue()
        semaphore = asyncio.Semaphore(1)

        async def failing_upsert(batch: list[int]) -> None:
            raise RuntimeError("boom")

        stage = PersistenceStage(
            client=_FakeQdrantClient(),
            collection_name="media-items",
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            persistence_queue=persistence_queue,
            retry_queue=retry_queue,
            upsert_semaphore=semaphore,
            upsert_buffer_size=5,
            upsert_fn=failing_upsert,
            on_batch_complete=None,
        )

        worker = asyncio.create_task(stage.run(0))

        await stage.enqueue_points([1])

        with pytest.raises(RuntimeError):
            await worker

        await asyncio.wait_for(persistence_queue.join(), timeout=1)

        return semaphore._value  # type: ignore[attr-defined]

    semaphore_value = asyncio.run(scenario())

    assert semaphore_value == 1


def test_persistence_stage_flushes_retry_queue_before_exit() -> None:
    async def scenario() -> tuple[list[list[int]], bool, bool]:
        persistence_queue: PersistenceQueue = asyncio.Queue()
        retry_queue: asyncio.Queue[list[list[int]]] = asyncio.Queue()
        semaphore = asyncio.Semaphore(1)

        processed: list[list[int]] = []

        async def upsert(batch: list[int]) -> None:
            processed.append(list(batch))

        stage = PersistenceStage(
            client=_FakeQdrantClient(),
            collection_name="media-items",
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            persistence_queue=persistence_queue,
            retry_queue=retry_queue,
            upsert_semaphore=semaphore,
            upsert_buffer_size=5,
            upsert_fn=upsert,
            on_batch_complete=None,
        )

        worker = asyncio.create_task(stage.run(0))

        await retry_queue.put([1, 2, 3])
        await persistence_queue.put(PERSIST_DONE)

        await asyncio.wait_for(worker, timeout=1)

        return processed, persistence_queue.empty(), retry_queue.empty()

    processed, persistence_empty, retry_empty = asyncio.run(scenario())

    assert processed == [[1, 2, 3]]
    assert persistence_empty
    assert retry_empty


def test_persistence_stage_leaves_no_lingering_queue_items() -> None:
    async def scenario() -> tuple[int, int]:
        persistence_queue: PersistenceQueue = asyncio.Queue()
        retry_queue: asyncio.Queue[list[int]] = asyncio.Queue()
        semaphore = asyncio.Semaphore(2)

        async def upsert(batch: list[int]) -> None:
            await asyncio.sleep(0)

        stage = PersistenceStage(
            client=_FakeQdrantClient(),
            collection_name="media-items",
            dense_vector_name="dense",
            sparse_vector_name="sparse",
            persistence_queue=persistence_queue,
            retry_queue=retry_queue,
            upsert_semaphore=semaphore,
            upsert_buffer_size=2,
            upsert_fn=upsert,
            on_batch_complete=None,
        )

        worker = asyncio.create_task(stage.run(0))

        await stage.enqueue_points([1, 2, 3, 4])
        await asyncio.wait_for(persistence_queue.join(), timeout=1)
        await persistence_queue.put(PERSIST_DONE)

        await asyncio.wait_for(worker, timeout=1)

        return persistence_queue.qsize(), retry_queue.qsize()

    persistence_remaining, retry_remaining = asyncio.run(scenario())

    assert persistence_remaining == 0
    assert retry_remaining == 0
