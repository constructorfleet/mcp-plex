import asyncio
from pathlib import Path
from types import SimpleNamespace

from mcp_plex import loader
from qdrant_client import models


class DummyTextEmbedding:
    def __init__(self, name: str):
        self.embedding_size = 3

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

    def passage_embed(self, texts):
        for i, _ in enumerate(texts):
            yield DummySparseVector([i], [1.0])


class DummyQdrantClient:
    instance = None

    def __init__(self, url: str, api_key: str | None = None):
        self.collections = {}
        self.upserted = []
        DummyQdrantClient.instance = self

    async def collection_exists(self, name: str) -> bool:
        return name in self.collections

    async def get_collection(self, name: str):
        return self.collections[name]

    async def delete_collection(self, name: str):
        self.collections.pop(name, None)

    async def create_collection(self, collection_name: str, vectors_config, sparse_vectors_config):
        size = vectors_config["dense"].size
        params = SimpleNamespace(vectors={"dense": models.VectorParams(size=size, distance=models.Distance.COSINE)})
        self.collections[collection_name] = SimpleNamespace(config=SimpleNamespace(params=params))

    async def create_payload_index(self, **kwargs):
        return None

    async def upsert(self, collection_name: str, points):
        self.upserted.extend(points)


async def _run_loader(sample_dir: Path):
    await loader.run(None, None, None, sample_dir, None, None)


def test_run_writes_points(monkeypatch):
    monkeypatch.setattr(loader, "TextEmbedding", DummyTextEmbedding)
    monkeypatch.setattr(loader, "SparseTextEmbedding", DummySparseEmbedding)
    monkeypatch.setattr(loader, "AsyncQdrantClient", DummyQdrantClient)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    asyncio.run(_run_loader(sample_dir))
    client = DummyQdrantClient.instance
    assert client is not None
    assert len(client.upserted) == 2
    payloads = [p.payload for p in client.upserted]
    assert all("title" in p and "type" in p for p in payloads)
