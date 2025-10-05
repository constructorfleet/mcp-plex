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
    assert "Storage worker" in caplog.text
    assert "handling 2 points" in caplog.text
    assert "processed 2 items" in caplog.text


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


def test_run_limits_concurrent_upserts(monkeypatch):
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyClient)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    monkeypatch.setattr(loader, "_qdrant_max_concurrent_upserts", 1)

    concurrency = {"current": 0, "max": 0}
    started = asyncio.Queue()
    release_queue = asyncio.Queue()
    third_requested = asyncio.Event()
    base_items = list(loader._load_from_sample(sample_dir))

    monkeypatch.setattr(
        loader,
        "_load_from_sample",
        lambda _: base_items + base_items[:1],
    )

    upsert_calls = {"count": 0}

    async def fake_upsert(client, collection_name, points, **kwargs):
        upsert_calls["count"] += 1
        if upsert_calls["count"] == 3:
            third_requested.set()
        concurrency["current"] += 1
        concurrency["max"] = max(concurrency["max"], concurrency["current"])
        await started.put(upsert_calls["count"])
        await release_queue.get()
        concurrency["current"] -= 1

    monkeypatch.setattr(loader, "_upsert_in_batches", fake_upsert)

    async def invoke():
        run_task = asyncio.create_task(
            loader.run(None, None, None, sample_dir, None, None, upsert_buffer_size=1)
        )
        await asyncio.wait_for(started.get(), timeout=1)
        assert not third_requested.is_set()
        await release_queue.put(None)
        await asyncio.wait_for(started.get(), timeout=1)
        assert not third_requested.is_set()
        await release_queue.put(None)
        await asyncio.wait_for(started.get(), timeout=1)
        await release_queue.put(None)
        await run_task

    asyncio.run(invoke())

    assert concurrency["max"] == 1
    assert third_requested.is_set()


def test_run_ensures_collection_before_loading(monkeypatch):
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyClient)
    order: list[str] = []

    async def fake_ensure(*args, **kwargs):
        order.append("ensure")

    monkeypatch.setattr(loader, "_ensure_collection", fake_ensure)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    original_execute = loader.LoaderPipeline.execute

    async def fake_execute(self):
        order.append("execute")
        self._items = []

    monkeypatch.setattr(loader.LoaderPipeline, "execute", fake_execute)

    asyncio.run(loader.run(None, None, None, sample_dir, None, None))

    assert order and order[0] == "ensure"
    assert "execute" in order

    monkeypatch.setattr(loader.LoaderPipeline, "execute", original_execute)
