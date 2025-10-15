import asyncio
import logging
from pathlib import Path

import pytest
from click.testing import CliRunner

from mcp_plex import loader
from mcp_plex.loader import samples as loader_samples
from mcp_plex.loader import cli as loader_cli
from qdrant_client import models


class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

    def _get_model_params(self, model_name):
        return 1, models.Distance.COSINE

    async def close(self):
        """Match the AsyncQdrantClient interface used by the loader."""
        return None

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
    with caplog.at_level(logging.INFO):
        asyncio.run(loader.run(None, None, None, sample_dir, None, None))
    assert "Starting staged loader (sample mode)" in caplog.text
    assert "Launching loader orchestrator" in caplog.text
    assert "Starting ingestion stage (sample mode)" in caplog.text
    assert "Upsert worker 0 handling 2 points" in caplog.text
    assert "Upsert worker 0 processed 2 items" in caplog.text
    assert "Loaded 2 items" in caplog.text
    assert "Ingestion stage finished" in caplog.text
    assert "Loader orchestrator run completed successfully" in caplog.text


def test_run_logs_no_points(monkeypatch, caplog):
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyClient)
    monkeypatch.setattr(loader_samples, "_load_from_sample", lambda _: [])
    monkeypatch.setattr(loader, "_load_from_sample", loader_samples._load_from_sample)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    with caplog.at_level(logging.INFO):
        asyncio.run(loader.run(None, None, None, sample_dir, None, None))
    assert "Loaded 0 items" in caplog.text
    assert "No points to upsert" in caplog.text
    assert "Ingestion stage finished" in caplog.text


def test_run_logs_qdrant_retry_summary(monkeypatch, caplog):
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyClient)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"

    async def fake_process_retry_queue(*args, **kwargs):
        return 7, 3

    monkeypatch.setattr(loader, "_process_qdrant_retry_queue", fake_process_retry_queue)

    with caplog.at_level(logging.INFO):
        asyncio.run(loader.run(None, None, None, sample_dir, None, None))

    summary_records = [
        record
        for record in caplog.records
        if record.levelno == logging.INFO
        and getattr(record, "event", None) == "qdrant_retry_summary"
    ]
    assert summary_records, "Expected a qdrant retry summary log record"
    record = summary_records[-1]
    assert getattr(record, "succeeded_points", None) == 7
    assert getattr(record, "failed_points", None) == 3


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

    concurrency = {"current": 0, "max": 0}
    started = asyncio.Queue()
    release_queue = asyncio.Queue()
    third_requested = asyncio.Event()
    base_items = list(loader_samples._load_from_sample(sample_dir))

    monkeypatch.setattr(
        loader_samples,
        "_load_from_sample",
        lambda _: base_items + base_items[:1],
    )
    monkeypatch.setattr(loader, "_load_from_sample", loader_samples._load_from_sample)

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
            loader.run(
                None,
                None,
                None,
                sample_dir,
                None,
                None,
                upsert_buffer_size=1,
                max_concurrent_upserts=1,
            )
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

    async def fake_run(self):
        order.append("execute")

    monkeypatch.setattr(loader.LoaderOrchestrator, "run", fake_run)

    asyncio.run(loader.run(None, None, None, sample_dir, None, None))

    assert order and order[0] == "ensure"
    assert "execute" in order


def test_main_accepts_log_level(monkeypatch, tmp_path):
    runner = CliRunner()
    configured: dict[str, int] = {}

    def fake_basic_config(**kwargs):
        configured["level"] = kwargs.get("level")

    def fake_asyncio_run(coro):
        coro.close()
        return None

    monkeypatch.setattr(loader_cli.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(loader_cli.asyncio, "run", fake_asyncio_run)

    sample_dir = tmp_path / "samples"
    sample_dir.mkdir()

    result = runner.invoke(
        loader_cli.main,
        [
            "--plex-url",
            "http://example.com",
            "--plex-token",
            "token",
            "--tmdb-api-key",
            "key",
            "--qdrant-url",
            "http://qdrant",
            "--sample-dir",
            str(sample_dir),
            "--log-level",
            "debug",
        ],
    )

    assert result.exit_code == 0
    assert configured.get("level") == logging.DEBUG
