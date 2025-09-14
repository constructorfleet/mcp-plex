from __future__ import annotations

import asyncio
import json
from pathlib import Path

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client import models

from mcp_plex import loader


class CaptureClient(AsyncQdrantClient):
    instance: "CaptureClient" | None = None
    captured_points: list[models.PointStruct] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CaptureClient.instance = self

    async def upsert(self, collection_name: str, points, **kwargs):
        CaptureClient.captured_points = points
        return await super().upsert(collection_name=collection_name, points=points, **kwargs)


async def _run_loader(sample_dir: Path) -> None:
    await loader.run(
        None,
        None,
        None,
        sample_dir,
        None,
        None,
    )


def test_run_writes_points(monkeypatch):
    monkeypatch.setattr(loader, "AsyncQdrantClient", CaptureClient)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    asyncio.run(_run_loader(sample_dir))
    client = CaptureClient.instance
    assert client is not None
    points, _ = asyncio.run(client.scroll("media-items", limit=10, with_payload=True))
    assert len(points) == 2
    assert all("title" in p.payload and "type" in p.payload for p in points)
    captured = CaptureClient.captured_points
    assert len(captured) == 2
    assert all(isinstance(p.vector["dense"], models.Document) for p in captured)
    assert all(p.vector["dense"].model == "BAAI/bge-small-en-v1.5" for p in captured)
    assert all(isinstance(p.vector["sparse"], models.Document) for p in captured)
    assert all(
        p.vector["sparse"].model == "Qdrant/bm42-all-minilm-l6-v2-attentions"
        for p in captured
    )


def test_run_processes_imdb_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "AsyncQdrantClient", CaptureClient)
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(json.dumps(["tt0111161"]))
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"

    async def fake_fetch(client, imdb_id):
        return None

    monkeypatch.setattr(loader, "_fetch_imdb", fake_fetch)

    asyncio.run(
        loader.run(
            None,
            None,
            None,
            sample_dir,
            None,
            None,
            imdb_queue_path=queue_file,
        )
    )

    assert json.loads(queue_file.read_text()) == ["tt0111161"]
