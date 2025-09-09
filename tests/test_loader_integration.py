from __future__ import annotations

import asyncio
from pathlib import Path

from qdrant_client.async_qdrant_client import AsyncQdrantClient

from mcp_plex import loader


class CaptureClient(AsyncQdrantClient):
    instance: "CaptureClient" | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CaptureClient.instance = self


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


