import asyncio
import builtins
import importlib
import json
import types
from datetime import datetime
from pathlib import Path

import httpx
from qdrant_client import models
import pytest

from mcp_plex import loader
from mcp_plex.loader.imdb_cache import IMDbCache
from mcp_plex.loader import (
    _fetch_imdb,
    _load_from_sample,
    _load_imdb_retry_queue,
    _persist_imdb_retry_queue,
    _process_imdb_retry_queue,
    _resolve_dense_model_params,
    build_point,
)
from mcp_plex.common.types import (
    AggregatedItem,
    IMDbName,
    IMDbRating,
    IMDbTitle,
    PlexGuid,
    PlexItem,
    PlexPerson,
    TMDBMovie,
)


def test_loader_import_fallback(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("plexapi"):
            raise ModuleNotFoundError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = importlib.reload(loader)
    assert module.PlexServer is None
    assert module.PlexPartialObject is object
    importlib.reload(loader)
def test_load_from_sample_returns_items():
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    items = _load_from_sample(sample_dir)
    assert len(items) == 2
    assert {i.plex.type for i in items} == {"movie", "episode"}


def test_fetch_imdb_cache_miss(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(loader, "_imdb_cache", IMDbCache(cache_path))

    calls = 0

    async def imdb_mock(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"id": "tt1", "type": "movie", "primaryTitle": "T"}
        )

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(imdb_mock)) as client:
            result = await _fetch_imdb(client, "tt1")
            assert result is not None

    asyncio.run(main())
    assert calls == 1
    data = json.loads(cache_path.read_text())
    assert data["tt1"]["id"] == "tt1"


def test_fetch_imdb_cache_hit(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps({"tt1": {"id": "tt1", "type": "movie", "primaryTitle": "T"}})
    )
    monkeypatch.setattr(loader, "_imdb_cache", IMDbCache(cache_path))

    async def error_mock(request):
        raise AssertionError("network should not be called")

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(error_mock)) as client:
            result = await _fetch_imdb(client, "tt1")
            assert result is not None
            assert result.id == "tt1"

    asyncio.run(main())


def test_fetch_imdb_retries_on_429(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(loader, "_imdb_cache", IMDbCache(cache_path))
    monkeypatch.setattr(loader, "_imdb_max_retries", 5)
    monkeypatch.setattr(loader, "_imdb_backoff", 0.1)

    call_count = 0

    async def imdb_mock(request):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(429)
        return httpx.Response(
            200, json={"id": "tt1", "type": "movie", "primaryTitle": "T"}
        )

    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(imdb_mock)) as client:
            result = await _fetch_imdb(client, "tt1")
            assert result is not None

    asyncio.run(main())
    assert call_count == 3
    assert delays == [0.1, 0.2]


def test_imdb_retry_queue_persists_and_retries(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    queue_path = tmp_path / "queue.json"
    monkeypatch.setattr(loader, "_imdb_cache", IMDbCache(cache_path))
    monkeypatch.setattr(loader, "_imdb_max_retries", 0)
    monkeypatch.setattr(loader, "_imdb_backoff", 0)

    async def first_transport(request):
        return httpx.Response(429)

    async def second_transport(request):
        return httpx.Response(
            200,
            json={
                "id": "tt0111161",
                "type": "movie",
                "primaryTitle": "The Shawshank Redemption",
            },
        )

    async def first_run():
        _load_imdb_retry_queue(queue_path)
        async with httpx.AsyncClient(transport=httpx.MockTransport(first_transport)) as client:
            await _process_imdb_retry_queue(client)
            await _fetch_imdb(client, "tt0111161")
        _persist_imdb_retry_queue(queue_path)

    asyncio.run(first_run())
    assert json.loads(queue_path.read_text()) == ["tt0111161"]

    async def second_run():
        _load_imdb_retry_queue(queue_path)
        assert loader._imdb_retry_queue is not None
        assert loader._imdb_retry_queue.qsize() == 1
        assert loader._imdb_retry_queue.snapshot() == ["tt0111161"]
        async with httpx.AsyncClient(transport=httpx.MockTransport(second_transport)) as client:
            await _process_imdb_retry_queue(client)
        _persist_imdb_retry_queue(queue_path)

    asyncio.run(second_run())
    assert json.loads(queue_path.read_text()) == []
    assert loader._imdb_cache.get("tt0111161") is not None


def test_load_imdb_retry_queue_invalid_json(tmp_path):
    path = tmp_path / "queue.json"
    path.write_text("not json")
    _load_imdb_retry_queue(path)
    assert loader._imdb_retry_queue is not None
    assert loader._imdb_retry_queue.qsize() == 0


def test_process_imdb_retry_queue_requeues(monkeypatch):
    queue = loader._IMDbRetryQueue(["tt0111161"])
    monkeypatch.setattr(loader, "_imdb_retry_queue", queue)

    async def fake_fetch(client, imdb_id):
        return None

    monkeypatch.setattr(loader, "_fetch_imdb", fake_fetch)

    async def run_test():
        async with httpx.AsyncClient() as client:
            await _process_imdb_retry_queue(client)

    asyncio.run(run_test())
    assert queue.qsize() == 1
    assert queue.snapshot() == ["tt0111161"]
def test_upsert_in_batches_handles_errors(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = 0

        async def upsert(self, collection_name: str, points, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise httpx.ConnectError("fail", request=httpx.Request("POST", ""))

    client = DummyClient()
    points = [models.PointStruct(id=i, vector={}, payload={}) for i in range(3)]
    monkeypatch.setattr(loader, "_qdrant_batch_size", 1)
    asyncio.run(loader._upsert_in_batches(client, "c", points))
    assert client.calls == 3


def test_upsert_in_batches_enqueues_retry_batches(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = 0

        async def upsert(self, collection_name: str, points, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadTimeout("timeout", request=httpx.Request("POST", ""))

    client = DummyClient()
    points = [models.PointStruct(id=i, vector={}, payload={}) for i in range(2)]
    retry_queue: asyncio.Queue[list[models.PointStruct]] = asyncio.Queue()
    monkeypatch.setattr(loader, "_qdrant_batch_size", 1)

    async def main() -> None:
        await loader._upsert_in_batches(
            client,
            "collection",
            points,
            retry_queue=retry_queue,
        )

    asyncio.run(main())
    assert retry_queue.qsize() == 1


def test_process_qdrant_retry_queue_retries_batches(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = 0

        async def upsert(self, collection_name: str, points, **kwargs):
            self.calls += 1
            if self.calls < 2:
                raise httpx.ReadTimeout("timeout", request=httpx.Request("POST", ""))

    client = DummyClient()
    retry_queue: asyncio.Queue[list[models.PointStruct]] = asyncio.Queue()

    async def main() -> None:
        retry_queue.put_nowait(
            [models.PointStruct(id=1, vector={}, payload={})]
        )
        monkeypatch.setattr(loader, "_qdrant_retry_attempts", 2)
        monkeypatch.setattr(loader, "_qdrant_retry_backoff", 0.01)

        sleeps: list[float] = []

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(loader.asyncio, "sleep", fake_sleep)

        await loader._process_qdrant_retry_queue(client, "collection", retry_queue)
        assert sleeps == [0.02]

    asyncio.run(main())
    assert client.calls == 2
    assert retry_queue.empty()


def test_resolve_dense_model_params_known_model():
    size, distance = _resolve_dense_model_params("BAAI/bge-small-en-v1.5")
    assert size == 384
    assert distance is models.Distance.COSINE


def test_resolve_dense_model_params_unknown_model():
    with pytest.raises(ValueError, match="Unknown dense embedding model"):
        _resolve_dense_model_params("not-a-real/model")


def test_imdb_retry_queue_desync_errors():
    queue = loader._IMDbRetryQueue(["tt1"])
    queue._items.clear()
    with pytest.raises(RuntimeError, match="Queue is not empty"):
        queue.get_nowait()

    queue = loader._IMDbRetryQueue(["tt2"])
    queue._queue.clear()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError, match="asyncio.Queue is empty"):
        queue.get_nowait()


def test_persist_imdb_retry_queue_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "_imdb_retry_queue", None)
    path = tmp_path / "retry.json"
    loader._persist_imdb_retry_queue(path)
    assert not path.exists()


def test_ensure_collection_skips_existing():
    class DummyClient:
        async def collection_exists(self, collection_name: str) -> bool:
            return True

        async def create_collection(self, *args, **kwargs):
            raise AssertionError("should not create collection")

        async def create_payload_index(self, *args, **kwargs):
            raise AssertionError("should not create index")

    asyncio.run(
        loader._ensure_collection(
            DummyClient(),
            "media-items",
            dense_size=1,
            dense_distance=models.Distance.COSINE,
        )
    )


def test_build_point_includes_metadata():
    plex_item = PlexItem(
        rating_key="1",
        guid="guid",
        type="movie",
        title="Sample",
        summary="Summary",
        year=2024,
        added_at=datetime.fromtimestamp(1),
        guids=[PlexGuid(id="plex://1")],
        tagline="Tagline",
        directors=[PlexPerson(id=1, tag="Director")],
        writers=[PlexPerson(id=2, tag="Writer")],
        actors=[PlexPerson(id=3, tag="Actor")],
        genres=["Action"],
        collections=["Favorites"],
    )
    imdb_title = IMDbTitle(
        id="tt1",
        type="movie",
        primaryTitle="Sample",
        plot="Plot",
        rating=IMDbRating(aggregateRating=8.0),
        directors=[IMDbName(id="nm1", displayName="Director")],
    )
    tmdb_movie = TMDBMovie(
        id=1,
        title="Sample",
        overview="Overview",
        tagline="Another tagline",
        reviews=[{"content": "Great"}],
    )
    item = AggregatedItem(plex=plex_item, imdb=imdb_title, tmdb=tmdb_movie)

    point = build_point(
        item,
        dense_model_name="BAAI/bge-small-en-v1.5",
        sparse_model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
    )

    assert point.payload["title"] == "Sample"
    assert point.payload["collections"] == ["Favorites"]
    assert point.payload["plot"] == "Plot"
    assert "Directed by" in point.vector["dense"].text
    assert point.vector["dense"].model == "BAAI/bge-small-en-v1.5"
    assert point.vector["sparse"].model == "Qdrant/bm42-all-minilm-l6-v2-attentions"


def test_loader_pipeline_processes_sample_batches(monkeypatch):
    sample_items = _load_from_sample(
        Path(__file__).resolve().parents[1] / "sample-data"
    )

    recorded_batches: list[list[models.PointStruct]] = []

    async def record_upsert(client, collection_name: str, points, **kwargs):
        recorded_batches.append(list(points))

    monkeypatch.setattr(loader, "_upsert_in_batches", record_upsert)

    orchestrator, processed_items, _ = loader._build_loader_orchestrator(
        client=object(),
        collection_name="media-items",
        dense_model_name="BAAI/bge-small-en-v1.5",
        sparse_model_name="Qdrant/bm42-all-minilm-l6-v2-attentions",
        tmdb_api_key=None,
        sample_items=sample_items,
        plex_server=None,
        plex_chunk_size=10,
        enrichment_batch_size=1,
        enrichment_workers=2,
        upsert_buffer_size=1,
        max_concurrent_upserts=1,
    )

    asyncio.run(orchestrator.run())

    assert len(processed_items) == len(sample_items)
    assert recorded_batches, "expected pipeline to emit upsert batches"
    payload = recorded_batches[0][0].payload
    assert payload["title"]
    assert payload["type"] in {"movie", "episode"}
    reviews = payload.get("reviews")
    if reviews is not None:
        assert isinstance(reviews, list)
