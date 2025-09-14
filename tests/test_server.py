from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import builtins
import pytest
from starlette.testclient import TestClient

from mcp_plex import loader


def _load_server(monkeypatch):
    from qdrant_client import async_qdrant_client

    class SharedClient(async_qdrant_client.AsyncQdrantClient):
        _instance: "SharedClient" | None = None
        _initialized = False

        def __new__(cls, *args, **kwargs):
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance

        def __init__(self, *args, **kwargs):
            if self.__class__._initialized:
                return
            super().__init__(*args, **kwargs)
            self.__class__._initialized = True

    monkeypatch.setattr(loader, "AsyncQdrantClient", SharedClient)
    monkeypatch.setattr(async_qdrant_client, "AsyncQdrantClient", SharedClient)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    asyncio.run(loader.run(None, None, None, sample_dir, None, None))
    return importlib.reload(importlib.import_module("mcp_plex.server"))


def test_qdrant_env_config(monkeypatch):
    from qdrant_client import async_qdrant_client

    captured = {}

    class CaptureClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(async_qdrant_client, "AsyncQdrantClient", CaptureClient)
    monkeypatch.setenv("QDRANT_HOST", "example.com")
    monkeypatch.setenv("QDRANT_PORT", "1234")
    monkeypatch.setenv("QDRANT_GRPC_PORT", "5678")
    monkeypatch.setenv("QDRANT_PREFER_GRPC", "1")
    monkeypatch.setenv("QDRANT_HTTPS", "1")
    module = importlib.reload(importlib.import_module("mcp_plex.server"))

    assert captured["host"] == "example.com"
    assert captured["port"] == 1234
    assert captured["grpc_port"] == 5678
    assert captured["prefer_grpc"] is True
    assert captured["https"] is True
    assert hasattr(module.server, "qdrant_client")


def test_server_tools(monkeypatch):
    server = _load_server(monkeypatch)

    movie_id = "49915"
    res = asyncio.run(server.get_media.fn(identifier=movie_id))
    assert res and res[0]["plex"]["title"] == "The Gentlemen"

    res = asyncio.run(server.get_media.fn(identifier="tt8367814"))
    assert res and res[0]["plex"]["rating_key"] == movie_id

    poster = asyncio.run(server.media_poster.fn(identifier=movie_id))
    assert isinstance(poster, str) and "thumb" in poster
    assert server.server.cache.get_poster(movie_id) == poster

    art = asyncio.run(server.media_background.fn(identifier=movie_id))
    assert isinstance(art, str) and "art" in art
    assert server.server.cache.get_background(movie_id) == art

    item = json.loads(asyncio.run(server.media_item.fn(identifier=movie_id)))
    assert item["plex"]["rating_key"] == movie_id
    assert (
        server.server.cache.get_payload(movie_id)["plex"]["rating_key"] == movie_id
    )

    ids = json.loads(asyncio.run(server.media_ids.fn(identifier=movie_id)))
    assert ids["imdb"] == "tt8367814"

    res = asyncio.run(server.search_media.fn(query="Matthew McConaughey crime movie", limit=1))
    assert res and res[0]["plex"]["title"] == "The Gentlemen"

    rec = asyncio.run(server.recommend_media.fn(identifier=movie_id, limit=1))
    assert rec and rec[0]["plex"]["rating_key"] == "61960"

    assert asyncio.run(server.recommend_media.fn(identifier="0", limit=1)) == []

    with pytest.raises(ValueError):
        asyncio.run(server.media_item.fn(identifier="0"))
    with pytest.raises(ValueError):
        asyncio.run(server.media_ids.fn(identifier="0"))
    with pytest.raises(ValueError):
        asyncio.run(server.media_poster.fn(identifier="0"))
    with pytest.raises(ValueError):
        asyncio.run(server.media_background.fn(identifier="0"))


def test_new_media_tools(monkeypatch):
    server = _load_server(monkeypatch)

    movies = asyncio.run(server.new_movies.fn(limit=1))
    assert movies and movies[0]["plex"]["type"] == "movie"
    assert movies[0]["plex"]["added_at"] is not None

    shows = asyncio.run(server.new_shows.fn(limit=1))
    assert shows and shows[0]["plex"]["type"] == "episode"
    assert shows[0]["plex"]["added_at"] is not None


def test_actor_movies(monkeypatch):
    server = _load_server(monkeypatch)

    movies = asyncio.run(
        server.actor_movies.fn(actor="Matthew McConaughey", limit=1)
    )
    assert movies and movies[0]["plex"]["title"] == "The Gentlemen"

    none = asyncio.run(
        server.actor_movies.fn(
            actor="Matthew McConaughey", year_from=1990, year_to=1999
        )
    )
    assert none == []


def test_reranker_import_failure(monkeypatch):
    monkeypatch.setenv("USE_RERANKER", "1")
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    server = importlib.reload(importlib.import_module("mcp_plex.server"))
    assert server._reranker is None


def test_reranker_init_failure(monkeypatch):
    monkeypatch.setenv("USE_RERANKER", "1")
    st_module = types.ModuleType("sentence_transformers")

    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    st_module.CrossEncoder = Broken
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_module)
    server = importlib.reload(importlib.import_module("mcp_plex.server"))
    assert server._reranker is None


def test_rest_endpoints(monkeypatch):
    module = _load_server(monkeypatch)
    client = TestClient(module.server.http_app())

    resp = client.post("/rest/get-media", json={"identifier": "49915"})
    assert resp.status_code == 200
    assert resp.json()[0]["plex"]["rating_key"] == "49915"

    resp = client.post("/rest/prompt/media-info", json={"identifier": "49915"})
    assert resp.status_code == 200
    msg = resp.json()[0]
    assert msg["role"] == "user"
    assert "The Gentlemen" in msg["content"]["text"]

    resp = client.get("/rest/resource/media-ids/49915")
    assert resp.status_code == 200
    assert resp.json()["rating_key"] == "49915"

    spec = client.get("/openapi.json").json()
    get_media = spec["paths"]["/rest/get-media"]["post"]
    assert get_media["description"].startswith("Retrieve media items")
    params = {p["name"]: p for p in get_media["parameters"]}
    assert params["identifier"]["schema"]["description"].startswith("Rating key")
    assert "/rest/prompt/media-info" in spec["paths"]
    assert "/rest/resource/media-ids/{identifier}" in spec["paths"]

    resp = client.get("/rest")
    assert resp.status_code == 200
