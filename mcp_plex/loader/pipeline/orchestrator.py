"""Coordinating logic tying the loader pipeline stages together."""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Protocol

from .channels import IngestQueue, PersistenceQueue

LOGGER = logging.getLogger("mcp_plex.loader.orchestrator")


@dataclass(frozen=True, slots=True)
class _StageSpec:
    """Descriptor for a running pipeline stage."""

    role: str
    worker_id: int | None = None


class _StageFailure(Exception):
    """Wrapper exception capturing the originating stage failure."""

    def __init__(self, spec: _StageSpec, error: BaseException) -> None:
        super().__init__(str(error))
        self.spec = spec
        self.error = error


class IngestionStageProtocol(Protocol):
    async def run(self) -> None:
        ...


class EnrichmentStageProtocol(Protocol):
    async def run(self) -> None:
        ...


class PersistenceStageProtocol(Protocol):
    async def run(self, worker_id: int) -> None:
        ...


class LoaderOrchestrator:
    """Run the ingestion, enrichment, and persistence stages with supervision."""

    def __init__(
        self,
        *,
        ingestion_stage: IngestionStageProtocol,
        enrichment_stage: EnrichmentStageProtocol,
        persistence_stage: PersistenceStageProtocol,
        ingest_queue: IngestQueue,
        persistence_queue: PersistenceQueue,
        persistence_worker_count: int = 1,
        logger: logging.Logger | None = None,
    ) -> None:
        if persistence_worker_count <= 0:
            raise ValueError("persistence_worker_count must be positive")

        self._ingestion_stage = ingestion_stage
        self._enrichment_stage = enrichment_stage
        self._persistence_stage = persistence_stage
        self._ingest_queue = ingest_queue
        self._persistence_queue = persistence_queue
        self._persistence_worker_count = int(persistence_worker_count)
        self._logger = logger or LOGGER

    async def run(self) -> None:
        """Execute the configured pipeline stages concurrently."""

        self._logger.info(
            "Launching loader orchestrator with %d persistence worker(s).",
            self._persistence_worker_count,
        )
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(
                    self._run_stage(
                        _StageSpec(role="ingestion"),
                        getattr(self._ingestion_stage, "run"),
                    )
                )
                group.create_task(
                    self._run_stage(
                        _StageSpec(role="enrichment"),
                        getattr(self._enrichment_stage, "run"),
                    )
                )
                persistence_runner = getattr(self._persistence_stage, "run")
                for worker_id in range(self._persistence_worker_count):
                    group.create_task(
                        self._run_stage(
                            _StageSpec(role="persistence", worker_id=worker_id),
                            persistence_runner,
                            worker_id,
                        )
                    )
        except* _StageFailure as exc_group:
            failures = list(exc_group.exceptions)
            await self._handle_failures(failures)
            # Re-raise the first underlying error after cleanup so callers see the
            # original exception rather than the wrapper.
            raise failures[0].error
        else:
            self._logger.info("Loader orchestrator run completed successfully.")

    async def _run_stage(
        self,
        spec: _StageSpec,
        runner: Callable[..., Awaitable[object] | object],
        *args: object,
    ) -> None:
        """Execute *runner* and wrap unexpected exceptions with stage metadata."""

        stage_name = self._describe_stage(spec)
        self._logger.info("Starting %s.", stage_name)
        try:
            result = runner(*args)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            self._logger.debug("%s cancelled.", stage_name)
            raise
        except BaseException as exc:
            self._logger.debug(
                "%s raised %s", stage_name, exc, exc_info=exc
            )
            raise _StageFailure(spec, exc) from exc
        else:
            self._logger.info("%s completed successfully.", stage_name)

    async def _handle_failures(self, failures: list[_StageFailure]) -> None:
        """Log stage-specific failures and drain queues during cancellation."""

        if not failures:
            return

        roles = {failure.spec.role for failure in failures}
        if "ingestion" in roles:
            self._logger.warning(
                "Ingestion stage failed; cancelling enrichment and persistence tasks."
            )
        else:
            self._logger.warning(
                "Downstream stage failed; cancelling ingestion and related tasks."
            )

        for failure in failures:
            stage_name = self._describe_stage(failure.spec)
            self._logger.error(
                "%s failed: %s",
                stage_name,
                failure.error,
                exc_info=failure.error,
            )

        drained_ingest = self._drain_queue(self._ingest_queue)
        drained_persist = self._drain_queue(self._persistence_queue)
        if drained_ingest:
            self._logger.debug(
                "Drained %d item(s) from the ingest queue during cancellation.",
                drained_ingest,
            )
        if drained_persist:
            self._logger.debug(
                "Drained %d item(s) from the persistence queue during cancellation.",
                drained_persist,
            )

        # Yield to the event loop so cancelled tasks can finish cleanup before the
        # caller observes the exception.  This mirrors the behaviour expected by
        # the stage-specific tests which verify cancellation side-effects.
        await asyncio.sleep(0)

    def _drain_queue(self, queue: IngestQueue | PersistenceQueue) -> int:
        """Remove any queued items so cancellation does not leave stale work."""

        drained = 0
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                drained += 1
                try:
                    queue.task_done()
                except ValueError:  # Queue.join() not in use; ignore bookkeeping.
                    pass
        return drained

    def _describe_stage(self, spec: _StageSpec) -> str:
        """Return a human-friendly name for *spec*."""

        role = spec.role.capitalize()
        if spec.worker_id is None:
            return f"{role} stage"
        return f"{role} stage (worker {spec.worker_id})"


__all__ = ["LoaderOrchestrator"]
