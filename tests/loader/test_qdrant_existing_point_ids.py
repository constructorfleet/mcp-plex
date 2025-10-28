"""Tests for Qdrant helper utilities."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Sequence

from qdrant_client.http.exceptions import ResponseHandlingException

from mcp_plex.loader import qdrant as qdrant_module


class ChunkingQdrantClient:
    """Test double that enforces a maximum batch size for ``retrieve`` calls."""

    def __init__(self, max_batch_size: int) -> None:
        self.max_batch_size = max_batch_size
        self.calls: list[Sequence[int | str]] = []

    async def retrieve(
        self,
        *,
        collection_name: str,
        ids: Sequence[int | str],
        with_payload: bool,
    ) -> list[SimpleNamespace]:
        self.calls.append(ids)
        if len(ids) > self.max_batch_size:
            raise ResponseHandlingException(Exception("batch too large"))
        return [SimpleNamespace(id=value) for value in ids if int(value) % 2 == 0]


def test_existing_point_ids_chunks_large_batches() -> None:
    client = ChunkingQdrantClient(max_batch_size=10)
    ids = [str(idx) for idx in range(1, 21)] + ["8", "14"]

    result = asyncio.run(
        qdrant_module._existing_point_ids(
            client=client,
            collection_name="media-items",
            point_ids=ids,
        )
    )

    assert result == {"2", "4", "6", "8", "10", "12", "14", "16", "18", "20"}
    assert len(client.calls) == 3
    assert len(client.calls[0]) == len(set(ids))
    assert all(len(call) <= 10 for call in client.calls[1:])
