from typing import Any
import asyncio
from pathlib import Path
import importlib
import types

from mcp_plex import loader
from qdrant_client import models


class DummyTextEmbedding:
    def __init__(self, name: str):
        self.embedding_size = 3

    @staticmethod
    def list_supported_models():
        return ["dummy"]

    def embed(self, texts):
        for _ in texts:
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
        return DummySparseVector([0], [1.0])


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
        return list(self.store.values())[:limit]

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

    asyncio.run(_setup_db(tmp_path))
    server = importlib.reload(importlib.import_module("mcp_plex.server"))

    movie_id = "49915"
    res = asyncio.run(server.get_media.fn(identifier=movie_id))
    assert res and res[0]["plex"]["title"] == "The Gentlemen"

    res = asyncio.run(server.get_media.fn(identifier="tt8367814"))
    assert res and res[0]["plex"]["rating_key"] == movie_id

    res = asyncio.run(server.get_media.fn(identifier="The Gentlemen"))
    assert res and res[0]["plex"]["rating_key"] == movie_id

    res = asyncio.run(
        server.search_media.fn(query="Matthew McConaughey crime movie", limit=1)
    )
    assert res and res[0]["plex"]["title"] == "The Gentlemen"

    res = asyncio.run(server.recommend_media.fn(identifier=movie_id, limit=1))
    assert res and res[0]["plex"]["rating_key"] == "61960"

    # Unknown identifier should yield no recommendations
    res = asyncio.run(server.recommend_media.fn(identifier="0", limit=1))
    assert res == []

    # Exercise search path with an ID that doesn't exist
    asyncio.run(server._find_records("12345", limit=1))
