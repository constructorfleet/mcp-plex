import asyncio
from unittest.mock import AsyncMock

import pytest
import httpx
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from mcp_plex.loader.qdrant import QdrantManager


def test_ensure_collection_creates_missing_collection() -> None:
    client = AsyncMock()
    client.get_collection.side_effect = UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=b"missing",
        headers=httpx.Headers(),
    )
    manager = QdrantManager(client, "media")

    asyncio.run(
        manager.ensure_collection(vector_size=4, distance=models.Distance.COSINE)
    )

    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "media"
    vector_params = kwargs["vectors_config"]
    assert vector_params.size == 4
    assert vector_params.distance == models.Distance.COSINE


def test_ensure_collection_noop_when_exists() -> None:
    client = AsyncMock()
    client.get_collection.return_value = {"status": "ok"}
    manager = QdrantManager(client, "media")

    asyncio.run(
        manager.ensure_collection(vector_size=4, distance=models.Distance.COSINE)
    )

    client.create_collection.assert_not_awaited()


def test_ensure_collection_raises_unexpected_errors() -> None:
    client = AsyncMock()
    client.get_collection.side_effect = UnexpectedResponse(
        status_code=500,
        reason_phrase="Server Error",
        content=b"boom",
        headers=httpx.Headers(),
    )
    manager = QdrantManager(client, "media")

    with pytest.raises(UnexpectedResponse):
        asyncio.run(
            manager.ensure_collection(vector_size=4, distance=models.Distance.COSINE)
        )


def test_upsert_and_delete_collection_delegate_to_client() -> None:
    client = AsyncMock()
    manager = QdrantManager(client, "media")
    points = [
        models.PointStruct(
            id=1,
            vector=[0.1, 0.2],
            payload={"title": "Sample Movie", "year": 1999},
        )
    ]

    asyncio.run(manager.upsert_points(points))
    asyncio.run(manager.delete_collection())

    client.upsert.assert_awaited_once_with(collection_name="media", points=points)
    client.delete_collection.assert_awaited_once_with("media")
