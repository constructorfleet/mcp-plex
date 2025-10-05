import asyncio
import logging

import pytest

from mcp_plex.loader.pipeline.orchestrator import LoaderOrchestrator


class FailingIngestionStage:
    def __init__(self, queue: asyncio.Queue[object]) -> None:
        self.queue = queue

    async def run(self) -> None:
        await self.queue.put("batch-1")
        raise RuntimeError("ingestion boom")


class BlockingEnrichmentStage:
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()
        self._blocker = asyncio.Event()

    async def run(self) -> None:
        try:
            await self._blocker.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class BlockingPersistenceStage:
    def __init__(self, queue: asyncio.Queue[object]) -> None:
        self.queue = queue
        self.cancelled = asyncio.Event()

    async def run(self, worker_id: int) -> None:
        try:
            while True:
                await self.queue.get()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class BlockingIngestionStage:
    def __init__(self, queue: asyncio.Queue[object]) -> None:
        self.queue = queue
        self.cancelled = asyncio.Event()
        self._blocker = asyncio.Event()

    async def run(self) -> None:
        try:
            await self.queue.put("batch-1")
            await self.queue.put("batch-2")
            await self._blocker.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class SingleBatchEnrichmentStage:
    def __init__(
        self,
        ingest_queue: asyncio.Queue[object],
        persistence_queue: asyncio.Queue[object],
    ) -> None:
        self.ingest_queue = ingest_queue
        self.persistence_queue = persistence_queue
        self.cancelled = asyncio.Event()
        self._blocker = asyncio.Event()

    async def run(self) -> None:
        try:
            payload = await self.ingest_queue.get()
            self.ingest_queue.task_done()
            await self.persistence_queue.put(payload)
            await self._blocker.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class FailingPersistenceStage:
    def __init__(self, queue: asyncio.Queue[object]) -> None:
        self.queue = queue

    async def run(self, worker_id: int) -> None:
        payload = await self.queue.get()
        self.queue.task_done()
        raise RuntimeError(f"persistence boom {worker_id}: {payload}")


def _build_orchestrator(
    *,
    ingestion_stage: object,
    enrichment_stage: object,
    persistence_stage: object,
    ingest_queue: asyncio.Queue[object],
    persistence_queue: asyncio.Queue[object],
) -> LoaderOrchestrator:
    return LoaderOrchestrator(
        ingestion_stage=ingestion_stage,
        enrichment_stage=enrichment_stage,
        persistence_stage=persistence_stage,
        ingest_queue=ingest_queue,
        persistence_queue=persistence_queue,
        persistence_worker_count=1,
    )


def test_ingestion_failure_cancels_downstream(caplog: pytest.LogCaptureFixture) -> None:
    ingest_queue: asyncio.Queue[object] = asyncio.Queue()
    persistence_queue: asyncio.Queue[object] = asyncio.Queue()
    ingestion_stage = FailingIngestionStage(ingest_queue)
    enrichment_stage = BlockingEnrichmentStage()
    persistence_stage = BlockingPersistenceStage(persistence_queue)
    orchestrator = _build_orchestrator(
        ingestion_stage=ingestion_stage,
        enrichment_stage=enrichment_stage,
        persistence_stage=persistence_stage,
        ingest_queue=ingest_queue,
        persistence_queue=persistence_queue,
    )

    async def _run() -> None:
        await orchestrator.run()

    with caplog.at_level(logging.ERROR, logger="mcp_plex.loader.orchestrator"):
        with pytest.raises(RuntimeError, match="ingestion boom"):
            asyncio.run(_run())

    assert enrichment_stage.cancelled.is_set()
    assert persistence_stage.cancelled.is_set()
    assert ingest_queue.qsize() == 0
    assert persistence_queue.qsize() == 0
    error_messages = [record.getMessage() for record in caplog.records]
    assert any("Ingestion stage failed" in message for message in error_messages)


def test_persistence_failure_cancels_upstream(caplog: pytest.LogCaptureFixture) -> None:
    ingest_queue: asyncio.Queue[object] = asyncio.Queue()
    persistence_queue: asyncio.Queue[object] = asyncio.Queue()
    ingestion_stage = BlockingIngestionStage(ingest_queue)
    enrichment_stage = SingleBatchEnrichmentStage(ingest_queue, persistence_queue)
    persistence_stage = FailingPersistenceStage(persistence_queue)
    orchestrator = _build_orchestrator(
        ingestion_stage=ingestion_stage,
        enrichment_stage=enrichment_stage,
        persistence_stage=persistence_stage,
        ingest_queue=ingest_queue,
        persistence_queue=persistence_queue,
    )

    async def _run() -> None:
        await orchestrator.run()

    with caplog.at_level(logging.ERROR, logger="mcp_plex.loader.orchestrator"):
        with pytest.raises(RuntimeError, match="persistence boom"):
            asyncio.run(_run())

    assert ingestion_stage.cancelled.is_set()
    assert enrichment_stage.cancelled.is_set()
    assert ingest_queue.qsize() == 0
    assert persistence_queue.qsize() == 0
    error_messages = [record.getMessage() for record in caplog.records]
    assert any("Persistence stage" in message for message in error_messages)
