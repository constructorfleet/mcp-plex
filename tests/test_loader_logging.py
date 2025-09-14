import asyncio
import logging
from pathlib import Path

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
