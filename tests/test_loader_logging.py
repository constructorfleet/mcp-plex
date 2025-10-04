import asyncio
import logging
from pathlib import Path

import pytest

from mcp_plex import loader
from qdrant_client import models


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    def _get_model_params(self, model_name):
        return 1, models.Distance.COSINE

    async def collection_exists(self, name):
        return False

    async def create_collection(self, *args, **kwargs):
        pass

    async def create_payload_index(self, *args, **kwargs):
        pass

    async def upsert(self, *args, **kwargs):
        pass


def test_run_logs_upsert(monkeypatch, caplog):
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyClient)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    with caplog.at_level(logging.INFO, logger="mcp_plex.loader"):
        asyncio.run(loader.run(None, None, None, sample_dir, None, None))
    assert "Loaded 2 items" in caplog.text
    assert "Upserting 2 points" in caplog.text


def test_run_logs_no_points(monkeypatch, caplog):
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyClient)
    monkeypatch.setattr(loader, "_load_from_sample", lambda _: [])
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    with caplog.at_level(logging.INFO, logger="mcp_plex.loader"):
        asyncio.run(loader.run(None, None, None, sample_dir, None, None))
    assert "Loaded 0 items" in caplog.text
    assert "No points to upsert" in caplog.text


def test_run_rejects_invalid_upsert_buffer_size(monkeypatch):
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyClient)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"

    async def invoke():
        await loader.run(
            None,
            None,
            None,
            sample_dir,
            None,
            None,
            upsert_buffer_size=0,
        )

    with pytest.raises(ValueError, match="upsert_buffer_size must be positive"):
        asyncio.run(invoke())


def test_run_ensures_collection_before_loading(monkeypatch):
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyClient)
    order: list[str] = []

    async def fake_ensure(*args, **kwargs):
        order.append("ensure")

    orig_iter = loader._iter_from_sample

    async def fake_iter(sample_dir):
        order.append("iter")
        async for item in orig_iter(sample_dir):
            yield item

    async def fake_upsert(client, collection_name, points):
        return None

    monkeypatch.setattr(loader, "_ensure_collection", fake_ensure)
    monkeypatch.setattr(loader, "_iter_from_sample", fake_iter)
    monkeypatch.setattr(loader, "_upsert_in_batches", fake_upsert)

    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    asyncio.run(loader.run(None, None, None, sample_dir, None, None))

    assert order and order[0] == "ensure"
    assert order.count("iter") >= 1
