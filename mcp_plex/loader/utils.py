"""Shared utilities for the loader package."""
from __future__ import annotations

import asyncio
import inspect
import logging
from typing import AsyncIterator, Awaitable, Iterable, List, Sequence, TypeVar

from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

T = TypeVar("T")


_DENSE_MODEL_PARAMS: dict[str, tuple[int, models.Distance]] = {
    "BAAI/bge-small-en-v1.5": (384, models.Distance.COSINE),
    "BAAI/bge-base-en-v1.5": (768, models.Distance.COSINE),
    "BAAI/bge-large-en-v1.5": (1024, models.Distance.COSINE),
    "text-embedding-3-small": (1536, models.Distance.COSINE),
    "text-embedding-3-large": (3072, models.Distance.COSINE),
}


def require_positive(value: int, *, name: str) -> int:
    """Return *value* if positive, otherwise raise a ``ValueError``."""

    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def close_coroutines(tasks: Sequence[Awaitable[object]]) -> None:
    """Close coroutine objects to avoid unawaited warnings."""

    for task in tasks:
        if inspect.iscoroutine(task):
            task.close()


logger = logging.getLogger("mcp_plex.loader")


async def iter_gather_in_batches(
    tasks: Sequence[Awaitable[T]], batch_size: int
) -> AsyncIterator[T]:
    """Yield results from awaitable tasks in fixed-size batches."""

    try:
        require_positive(batch_size, name="batch_size")
    except ValueError:
        close_coroutines(tasks)
        raise

    total = len(tasks)
    for i in range(0, total, batch_size):
        batch = tasks[i : i + batch_size]
        for result in await asyncio.gather(*batch):
            yield result
        logger.info("Processed %d/%d items", min(i + batch_size, total), total)


def resolve_dense_model_params(model_name: str) -> tuple[int, models.Distance]:
    """Look up Qdrant vector parameters for a known dense embedding model."""

    try:
        return _DENSE_MODEL_PARAMS[model_name]
    except KeyError as exc:
        raise ValueError(
            "Unknown dense embedding model "
            f"'{model_name}'. Update _DENSE_MODEL_PARAMS with the model's size "
            "and distance."
        ) from exc


def is_local_qdrant(client: AsyncQdrantClient) -> bool:
    """Return ``True`` if *client* targets an in-process Qdrant instance."""

    inner = getattr(client, "_client", None)
    return bool(inner) and inner.__class__.__module__.startswith(
        "qdrant_client.local"
    )


async def gather_in_batches(
    tasks: Sequence[Awaitable[T]], batch_size: int
) -> List[T]:
    """Gather awaitable tasks in fixed-size batches."""

    return [result async for result in iter_gather_in_batches(tasks, batch_size)]
