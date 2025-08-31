import asyncio
import importlib
import json
import sys
import time
import types
from pathlib import Path
from typing import Any

import sys
import pytest
from qdrant_client import models

from mcp_plex import loader


class DummyTextEmbedding:
    def __init__(self, name: str):
        self.embedding_size = 3

    @staticmethod
    def list_supported_models():
        return ["dummy"]

    def embed(self, texts):
        for _ in texts:
            time.sleep(0.1)
            yield [0.1, 0.2, 0.3]


class DummyArray(list):
    def tolist(self):
        return list(self)


class DummySparseVector:
    def __init__(self, indices, values):
        self.indices = DummyArray(indices)
        self.values = DummyArray(values)


class DummySparseEmbedding:
    def __init__(self, name: str):
        pass

    @staticmethod
    def list_supported_models():
        return ["dummy"]

    def passage_embed(self, texts):
        for i, _ in enumerate(texts):
            yield DummySparseVector([i], [1.0])

    def query_embed(self, text):
        time.sleep(0.1)
        yield DummySparseVector([0], [1.0])


class DummyReranker:
    def __init__(self, model: str):
        pass

    def predict(self, pairs):
        scores = []
        for _, doc in pairs:
            if "Gentlemen" in doc:
                scores.append(10)
            elif "C" in doc:
                scores.append(2)
            elif "B" in doc:
                scores.append(1)
            else:
                scores.append(0)
        return scores


class DummyReranker:
    def __init__(self, model: str):
        pass

    def predict(self, pairs):
        scores = []
        for _, doc in pairs:
            if "Gentlemen" in doc:
                scores.append(10)
            elif "C" in doc:
                scores.append(2)
            elif "B" in doc:
                scores.append(1)
            else:
                scores.append(0)
        return scores


class DummyQdrantClient:
    store: dict[Any, models.Record] = {}
    size: int = 3

    def __init__(self, *args, **kwargs):
        DummyQdrantClient.instance = self

    async def collection_exists(self, name: str) -> bool:
        return True

    async def get_collection(self, name: str):
        params = types.SimpleNamespace(vectors={"dense": models.VectorParams(size=self.size, distance=models.Distance.COSINE)})
        return types.SimpleNamespace(config=types.SimpleNamespace(params=params))

    async def delete_collection(self, name: str):
        self.store.clear()

    async def create_collection(self, collection_name: str, vectors_config, sparse_vectors_config):
        DummyQdrantClient.size = vectors_config["dense"].size
        self.store.clear()

    async def create_payload_index(self, **kwargs):
        return None

    async def upsert(self, collection_name: str, points):
        for p in points:
            self.store[p.id] = p

    async def retrieve(self, collection_name: str, ids, with_payload=False):
        return [self.store[i] for i in ids if i in self.store]

    async def scroll(self, collection_name: str, limit: int, filter=None, with_payload=False):
        records = list(self.store.values())
        if filter and filter.should:
            def matches(rec, cond):
                value = rec.payload
                for part in cond.key.split('.'):
                    value = value.get(part) if isinstance(value, dict) else None
                if isinstance(cond.match, models.MatchValue):
                    return value == cond.match.value
                if isinstance(cond.match, models.MatchText):
                    return cond.match.text.lower() in str(value).lower()
                return False
            records = [r for r in records if any(matches(r, c) for c in filter.should)]
        return records[:limit], None

    async def search(self, collection_name: str, query_vector, query_sparse_vector=None, limit: int = 5, with_payload=False, **kwargs):
        records = list(self.store.values())[:limit]
        return [
            models.ScoredPoint(
                id=r.id, version=1, score=0.0, payload=r.payload, vector=None
            )
            for r in records
        ]

    async def recommend(self, collection_name: str, positive, limit: int = 5, with_payload=False, **kwargs):
        return [r for r in self.store.values() if r.id not in positive][:limit]


async def _setup_db(tmp_path: Path) -> str:
    await loader.run(
        plex_url=None,
        plex_token=None,
        tmdb_api_key=None,
        sample_dir=Path("sample-data"),
        qdrant_url="dummy",
        qdrant_api_key=None,
    )
    return "dummy"


def test_server_tools(tmp_path, monkeypatch):
    # Patch embeddings and Qdrant client to use dummy implementations
    monkeypatch.setattr(loader, "TextEmbedding", DummyTextEmbedding)
    monkeypatch.setattr(loader, "SparseTextEmbedding", DummySparseEmbedding)
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyQdrantClient)
    import fastembed
    from qdrant_client import async_qdrant_client
    monkeypatch.setattr(fastembed, "TextEmbedding", DummyTextEmbedding)
    monkeypatch.setattr(fastembed, "SparseTextEmbedding", DummySparseEmbedding)
    monkeypatch.setattr(async_qdrant_client, "AsyncQdrantClient", DummyQdrantClient)
    monkeypatch.setenv("USE_RERANKER", "1")
    st_module = types.ModuleType("sentence_transformers")
    st_module.CrossEncoder = DummyReranker
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_module)

    asyncio.run(_setup_db(tmp_path))
    server = importlib.reload(importlib.import_module("mcp_plex.server"))

    movie_id = "49915"
    res = asyncio.run(server.get_media.fn(identifier=movie_id))
    assert res and res[0]["plex"]["title"] == "The Gentlemen"

    res = asyncio.run(server.get_media.fn(identifier="tt8367814"))
    assert res and res[0]["plex"]["rating_key"] == movie_id

    res = asyncio.run(server.get_media.fn(identifier="The Gentlemen"))
    assert res and res[0]["plex"]["rating_key"] == movie_id

    poster = asyncio.run(server.media_poster.fn(identifier=movie_id))
    assert isinstance(poster, str) and "thumb" in poster
    art = asyncio.run(server.media_background.fn(identifier=movie_id))
    assert isinstance(art, str) and "art" in art
    item = json.loads(asyncio.run(server.media_item.fn(identifier=movie_id)))
    assert item["plex"]["rating_key"] == movie_id

    start = time.perf_counter()
    res = asyncio.run(
        server.search_media.fn(query="Matthew McConaughey crime movie", limit=1)
    )
    elapsed = time.perf_counter() - start
    assert elapsed < 0.2
    assert res and res[0]["plex"]["title"] == "The Gentlemen"

    # _find_records should handle client retrieval errors gracefully
    orig_retrieve, orig_scroll = server._client.retrieve, server._client.scroll
    async def fail(*args, **kwargs):
        raise AssertionError("client called")

    server._client.retrieve = fail
    asyncio.run(server._find_records("12345", limit=1))
    server._client.retrieve = orig_retrieve

    # Prefetched payloads should allow resource access without hitting the client
    server._client.retrieve = fail
    server._client.scroll = fail
    try:
        poster = asyncio.run(server.media_poster.fn(identifier=movie_id))
        assert isinstance(poster, str) and "thumb" in poster

        art = asyncio.run(server.media_background.fn(identifier=movie_id))
        assert isinstance(art, str) and "art" in art

        item = json.loads(asyncio.run(server.media_item.fn(identifier=movie_id)))
        assert item["plex"]["rating_key"] == movie_id

        ids = json.loads(asyncio.run(server.media_ids.fn(identifier=movie_id)))
        assert ids["imdb"] == "tt8367814"
    finally:
        server._client.retrieve = orig_retrieve
        server._client.scroll = orig_scroll

    with pytest.raises(AssertionError):
        asyncio.run(fail())

    monkeypatch.setattr(server, "_CACHE_SIZE", 1)
    server._cache_set(server._poster_cache, "a", "1")
    server._cache_set(server._poster_cache, "b", "2")

    # Reranking should reorder results based on cross-encoder scores
    orig_search = server._client.search
    async def fake_search(*args, **kwargs):
        return [
            models.ScoredPoint(id=1, version=1, score=0.0, payload={"data": {"title": "A"}}, vector=None),
            models.ScoredPoint(id=2, version=1, score=0.0, payload={"data": {"title": "B"}}, vector=None),
            models.ScoredPoint(id=3, version=1, score=0.0, payload={"data": {"title": "C"}}, vector=None),
        ]
    server._client.search = fake_search
    try:
        res = asyncio.run(server.search_media.fn(query="test", limit=2))
        assert [i["title"] for i in res] == ["C", "B"]
    finally:
        server._client.search = orig_search

    res = asyncio.run(server.recommend_media.fn(identifier=movie_id, limit=1))
    assert res and res[0]["plex"]["rating_key"] == "61960"

    # Unknown identifier should yield no recommendations
    res = asyncio.run(server.recommend_media.fn(identifier="0", limit=1))
    assert res == []

    # Exercise search path with an ID that doesn't exist
    asyncio.run(server._find_records("12345", limit=1))

    with pytest.raises(ValueError):
        asyncio.run(server.media_item.fn(identifier="0"))
    with pytest.raises(ValueError):
        asyncio.run(server.media_ids.fn(identifier="0"))

    with pytest.raises(ValueError):
        asyncio.run(server.media_poster.fn(identifier="0"))

    with pytest.raises(ValueError):
        asyncio.run(server.media_background.fn(identifier="0"))


def _patch_dependencies(monkeypatch):
    monkeypatch.setattr(loader, "TextEmbedding", DummyTextEmbedding)
    monkeypatch.setattr(loader, "SparseTextEmbedding", DummySparseEmbedding)
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyQdrantClient)
    import fastembed
    from qdrant_client import async_qdrant_client
    monkeypatch.setattr(fastembed, "TextEmbedding", DummyTextEmbedding)
    monkeypatch.setattr(fastembed, "SparseTextEmbedding", DummySparseEmbedding)
    monkeypatch.setattr(async_qdrant_client, "AsyncQdrantClient", DummyQdrantClient)


def test_reranker_import_failure(monkeypatch):
    _patch_dependencies(monkeypatch)
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    import builtins
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setenv("USE_RERANKER", "1")
    server = importlib.reload(importlib.import_module("mcp_plex.server"))

    async def fake_search(*args, **kwargs):
        return [
            models.ScoredPoint(id=1, version=1, score=0.0, payload={"data": {"title": "A", "plex": {"rating_key": 1}}}, vector=None),
            models.ScoredPoint(id=2, version=1, score=0.0, payload={"data": {"title": "B", "plex": {"rating_key": 2}}}, vector=None),
        ]

    server._client.search = fake_search
    res = asyncio.run(server.search_media.fn(query="test", limit=2))
    assert [i["title"] for i in res] == ["A", "B"]


def test_reranker_init_failure(monkeypatch):
    _patch_dependencies(monkeypatch)
    monkeypatch.setenv("USE_RERANKER", "1")
    st_module = types.ModuleType("sentence_transformers")

    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    st_module.CrossEncoder = Broken
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_module)
    server = importlib.reload(importlib.import_module("mcp_plex.server"))
    assert server._reranker is None

    async def fake_search(*args, **kwargs):
        return [
            models.ScoredPoint(id=1, version=1, score=0.0, payload={"data": {"title": "A", "plex": {"rating_key": 1}}}, vector=None),
            models.ScoredPoint(id=2, version=1, score=0.0, payload={"data": {"title": "B", "plex": {"rating_key": 2}}}, vector=None),
        ]

    server._client.search = fake_search
    res = asyncio.run(server.search_media.fn(query="test", limit=2))
    assert [i["title"] for i in res] == ["A", "B"]
