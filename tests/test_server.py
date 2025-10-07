from __future__ import annotations

import asyncio
import importlib
import json
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import builtins
from typing import Any

import pytest
from qdrant_client import models
from starlette.testclient import TestClient

from mcp_plex import loader
from mcp_plex import server as server_module
from mcp_plex.server import media as media_helpers
from mcp_plex.server.tools import media_library as media_library_tools
from pydantic import ValidationError


@contextmanager
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

        async def close(self) -> None:  # type: ignore[override]
            """Prevent sample data teardown from closing the shared instance."""
            return None

    monkeypatch.setattr(loader, "AsyncQdrantClient", SharedClient)
    monkeypatch.setattr(async_qdrant_client, "AsyncQdrantClient", SharedClient)
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    asyncio.run(loader.run(None, None, None, sample_dir, None, None))
    module = importlib.reload(importlib.import_module("mcp_plex.server"))
    try:
        yield module
    finally:
        asyncio.run(module.server.close())


def test_qdrant_env_config(monkeypatch):
    from qdrant_client import async_qdrant_client

    captured = {}

    class CaptureClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
        async def close(self):
            pass

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
    asyncio.run(module.server.close())


def test_server_tools(monkeypatch):
    with _load_server(monkeypatch) as server:
        assert callable(media_library_tools.register_media_library_tools)
        movie_id = "49915"
        res = asyncio.run(server.get_media.fn(identifier=movie_id))
        assert res and res[0]["plex"]["title"] == "The Gentlemen"

        res = asyncio.run(server.get_media.fn(identifier="tt8367814"))
        assert res and res[0]["plex"]["rating_key"] == movie_id

        episode = asyncio.run(server.get_media.fn(identifier="61960"))
        assert episode and episode[0]["show_title"] == "Alien: Earth"
        assert episode[0]["season_number"] == 1
        assert episode[0]["episode_number"] == 4

        poster = asyncio.run(server.media_poster.fn(identifier=movie_id))
        assert isinstance(poster, str) and "thumb" in poster
        assert server.server.cache.get_poster(movie_id) == poster

        art = asyncio.run(server.media_background.fn(identifier=movie_id))
        assert isinstance(art, str) and "art" in art
        assert server.server.cache.get_background(movie_id) == art

        item = json.loads(asyncio.run(server.media_item.fn(identifier=movie_id)))
        assert item["plex"]["rating_key"] == movie_id
        assert (
            server.server.cache.get_payload(movie_id)["plex"]["rating_key"]
            == movie_id
        )

        ids = json.loads(asyncio.run(server.media_ids.fn(identifier=movie_id)))
        assert ids["imdb"] == "tt8367814"

        res = asyncio.run(
            server.search_media.fn(query="Matthew McConaughey crime movie", limit=1)
        )
        assert res and res[0]["plex"]["title"] == "The Gentlemen"

        structured = asyncio.run(
            server.query_media.fn(
                dense_query="crime comedy",
                title="Gentlemen",
                type="movie",
                directors=["Guy Ritchie"],
                limit=1,
            )
        )
        assert structured and structured[0]["plex"]["title"] == "The Gentlemen"
        assert "directors" in structured[0]

        episode_structured = asyncio.run(
            server.query_media.fn(
                type="episode",
                show_title="Alien: Earth",
                season_number=1,
                episode_number=4,
                limit=1,
            )
        )
        assert episode_structured and episode_structured[0]["plex"]["rating_key"] == "61960"
        assert episode_structured[0]["show_title"] == "Alien: Earth"

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


def test_get_media_data_caches_external_ids(monkeypatch):
    with _load_server(monkeypatch) as server:
        call_count = 0

        original_find_records = media_helpers._find_records

        async def _counting_find_records(
            plex_server, identifier: str, limit: int = 1
        ):
            nonlocal call_count
            call_count += 1
            return await original_find_records(plex_server, identifier, limit=limit)

        monkeypatch.setattr(media_helpers, "_find_records", _counting_find_records)

        imdb_id = "tt8367814"
        tmdb_id = "522627"

        plex_server = server.server

        data = asyncio.run(media_helpers._get_media_data(plex_server, imdb_id))
        assert data["plex"]["rating_key"] == "49915"
        assert call_count == 1

        cached_imdb = asyncio.run(media_helpers._get_media_data(plex_server, imdb_id))
        assert cached_imdb["plex"]["rating_key"] == "49915"
        assert call_count == 1

        cached_tmdb = asyncio.run(media_helpers._get_media_data(plex_server, tmdb_id))
        assert cached_tmdb["plex"]["rating_key"] == "49915"
        assert call_count == 1


def test_new_media_tools(monkeypatch):
    with _load_server(monkeypatch) as server:
        movies = asyncio.run(server.new_movies.fn(limit=1))
        assert movies and movies[0]["plex"]["type"] == "movie"
        assert movies[0]["plex"]["added_at"] is not None

        shows = asyncio.run(server.new_shows.fn(limit=1))
        assert shows and shows[0]["plex"]["type"] == "episode"
        assert shows[0]["plex"]["added_at"] is not None
        assert shows[0]["show_title"] == "Alien: Earth"
        assert shows[0]["season_number"] == 1
        assert shows[0]["episode_number"] == 4


def test_actor_movies(monkeypatch):
    with _load_server(monkeypatch) as server:
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


def test_play_media_requires_configuration(monkeypatch):
    with _load_server(monkeypatch) as server:
        with pytest.raises(RuntimeError):
            asyncio.run(
                server.play_media.fn(identifier="49915", player="Living Room")
            )


def test_play_media_with_alias(monkeypatch):
    monkeypatch.setenv("PLEX_URL", "http://plex.test:32400")
    monkeypatch.setenv("PLEX_TOKEN", "token")
    monkeypatch.setenv(
        "PLEX_PLAYER_ALIASES",
        json.dumps(
            {
                "machine-123": ["Living Room", "Movie Room"],
                "client-abc": "Living Room",
                "machine-123:client-abc": ["Living Room"],
            }
        ),
    )

    class FakeMedia:
        def __init__(self, key: str) -> None:
            self.key = key

    play_requests: list[dict[str, Any]] = []
    fetch_requests: list[str] = []

    class FakeClient:
        def __init__(self) -> None:
            self.machineIdentifier = "machine-123"
            self.clientIdentifier = "client-abc"
            self.provides = "player,controller"
            self.address = "10.0.0.5"
            self.port = 32500
            self.product = "Plex for Apple TV"
            self.title = "Plex for Apple TV"

        def playMedia(self, media: FakeMedia, **kwargs: Any) -> None:
            play_requests.append({"media": media, "kwargs": kwargs})

    class FakePlex:
        def __init__(self, baseurl: str, token: str) -> None:
            assert baseurl.rstrip("/") == "http://plex.test:32400"
            assert token == "token"
            self.machineIdentifier = "server-001"
            self._client = FakeClient()

        def clients(self) -> list[FakeClient]:
            return [self._client]

        def fetchItem(self, key: str) -> FakeMedia:
            fetch_requests.append(key)
            return FakeMedia(key)

    with _load_server(monkeypatch) as server:
        monkeypatch.setattr(server, "PlexServerClient", FakePlex)

        result = asyncio.run(
            server.play_media.fn(identifier="49915", player="Living Room")
        )

        assert result["player"] == "Living Room"
        assert result["rating_key"] == "49915"
        assert fetch_requests == ["/library/metadata/49915"]
        assert play_requests, "Expected plexapi playMedia call"
        play_call = play_requests[0]
        assert isinstance(play_call["media"], FakeMedia)
        assert play_call["media"].key == "/library/metadata/49915"
        assert play_call["kwargs"]["machineIdentifier"] == "server-001"
        assert play_call["kwargs"]["offset"] == 0

        play_requests.clear()
        fetch_requests.clear()

        offset_seconds = 37
        offset_result = asyncio.run(
            server.play_media.fn(
                identifier="49915", player="Living Room", offset_seconds=offset_seconds
            )
        )

        assert offset_result["player"] == "Living Room"
        assert offset_result["rating_key"] == "49915"
        assert offset_result["offset_seconds"] == offset_seconds
        assert fetch_requests == ["/library/metadata/49915"]
        assert play_requests, "Expected plexapi playMedia call with offset"
        offset_call = play_requests[0]
        assert offset_call["kwargs"]["machineIdentifier"] == "server-001"
        assert offset_call["kwargs"]["offset"] == offset_seconds * 1000


def test_play_media_requires_player_capability(monkeypatch):
    monkeypatch.setenv("PLEX_URL", "http://plex.test:32400")
    monkeypatch.setenv("PLEX_TOKEN", "token")

    class FakeClient:
        def __init__(self) -> None:
            self.machineIdentifier = "machine-999"
            self.clientIdentifier = "client-999"
            self.provides = "controller"
            self.address = "10.0.0.10"
            self.port = 32500
            self.product = "Controller Only"
            self.title = "Controller Only"

        def playMedia(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("Playback should not be attempted")

    class FakePlex:
        def __init__(self, baseurl: str, token: str) -> None:
            assert baseurl.rstrip("/") == "http://plex.test:32400"
            assert token == "token"
            self.machineIdentifier = "server-001"
            self._client = FakeClient()

        def clients(self) -> list[FakeClient]:
            return [self._client]

        def fetchItem(self, key: str) -> Any:
            raise AssertionError("fetchItem should not be called")

    with _load_server(monkeypatch) as server:
        monkeypatch.setattr(server, "PlexServerClient", FakePlex)
        with pytest.raises(ValueError, match="cannot be controlled for playback"):
            asyncio.run(
                server.play_media.fn(identifier="49915", player="machine-999")
            )


def test_match_player_fuzzy_alias_resolution():
    players: list[server_module.PlexPlayerMetadata] = [
        {
            "display_name": "Movie Room TV",
            "name": "Plex for Apple TV",
            "product": "Apple TV",
            "machine_identifier": "machine-1",
            "client_identifier": "client-1",
            "friendly_names": ["Movie Room", "Movie Room TV"],
            "provides": {"player"},
            "client": None,
        },
        {
            "display_name": "Bedroom TV",
            "name": "Plex for Roku",
            "product": "Roku",
            "machine_identifier": "machine-2",
            "client_identifier": "client-2",
            "friendly_names": ["Bedroom"],
            "provides": {"player"},
            "client": None,
        },
    ]

    matched = server_module._match_player("movie rm", players)
    assert matched is players[0]


def test_match_player_unknown_raises():
    players: list[server_module.PlexPlayerMetadata] = [
        {
            "display_name": "Bedroom TV",
            "name": "Plex for Roku",
            "product": "Roku",
            "machine_identifier": "machine-2",
            "client_identifier": "client-2",
            "friendly_names": ["Bedroom"],
            "provides": {"player"},
            "client": None,
        }
    ]

    with pytest.raises(ValueError):
        server_module._match_player("Kitchen", players)


def test_match_player_whitespace_query_preserves_original_input():
    query = "   "

    with pytest.raises(ValueError) as exc:
        server_module._match_player(query, [])

    assert str(exc.value) == "Player '   ' not found"


def test_reranker_import_failure(monkeypatch, caplog):
    monkeypatch.setenv("USE_RERANKER", "1")
    orig_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with caplog.at_level(logging.WARNING, logger="mcp_plex.server"):
        module = importlib.reload(importlib.import_module("mcp_plex.server"))
    assert module.server.reranker is None
    assert any(
        "Failed to import CrossEncoder" in message
        for message in caplog.messages
    )
    asyncio.run(module.server.close())


def test_reranker_init_failure(monkeypatch, caplog):
    monkeypatch.setenv("USE_RERANKER", "1")
    st_module = types.ModuleType("sentence_transformers")

    class Broken:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom")

    st_module.CrossEncoder = Broken
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_module)
    module = importlib.reload(importlib.import_module("mcp_plex.server"))
    with caplog.at_level(logging.WARNING, logger="mcp_plex.server"):
        assert module.server.reranker is None
    assert any(
        "Failed to initialize CrossEncoder reranker" in message
        for message in caplog.messages
    )
    asyncio.run(module.server.close())


def test_rest_endpoints(monkeypatch):
    with _load_server(monkeypatch) as module:
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
        def _resolve(schema: dict):
            if "$ref" in schema:
                ref = schema["$ref"].split("/")[-1]
                return spec["components"]["schemas"][ref]
            return schema

        get_media = spec["paths"]["/rest/get-media"]["post"]
        assert get_media["description"].startswith("Retrieve media items")
        assert "parameters" not in get_media or not get_media["parameters"]
        get_media_schema = get_media["requestBody"]["content"]["application/json"][
            "schema"
        ]
        get_media_schema = _resolve(get_media_schema)
        assert (
            get_media_schema["properties"]["identifier"]["description"].startswith(
                "Rating key"
            )
        )

        search_media = spec["paths"]["/rest/search-media"]["post"]
        assert "parameters" not in search_media or not search_media["parameters"]
        search_schema = search_media["requestBody"]["content"][
            "application/json"
        ]["schema"]
        search_schema = _resolve(search_schema)
        assert "query" in search_schema["required"]
        assert "/rest/prompt/media-info" in spec["paths"]
        assert "/rest/resource/media-ids/{identifier}" in spec["paths"]

        resp = client.get("/rest")
        assert resp.status_code == 200


def test_server_lifespan_context(monkeypatch):
    with _load_server(monkeypatch) as module:
        closed = False

        async def fake_close() -> None:
            nonlocal closed
            closed = True

        monkeypatch.setattr(module.server, "close", fake_close)

        async def _lifespan() -> None:
            async with module.server._mcp_server.lifespan(module.server):
                pass

        asyncio.run(_lifespan())
        assert closed is True


def test_request_model_no_parameters():
    module = importlib.import_module("mcp_plex.server")

    async def _noop() -> None:
        return None

    assert module._request_model("noop", _noop) is None


def test_request_model_missing_annotation_uses_object():
    module = importlib.import_module("mcp_plex.server")

    async def _unannotated(foo):  # type: ignore[no-untyped-def]
        return foo

    request_model = module._request_model("unannotated", _unannotated)
    assert request_model is not None
    field = request_model.model_fields["foo"]
    assert field.annotation is object
    with pytest.raises(ValidationError):
        request_model()
    instance = request_model(foo="value")
    assert instance.foo == "value"


def test_normalize_identifier_scalar_inputs():
    assert media_helpers._normalize_identifier("  value  ") == "value"
    assert media_helpers._normalize_identifier(123) == "123"
    assert media_helpers._normalize_identifier(0.0) == "0.0"
    assert media_helpers._normalize_identifier("") is None
    assert media_helpers._normalize_identifier(None) is None


def test_run_config_to_kwargs():
    module = importlib.import_module("mcp_plex.server.cli")

    config = module.RunConfig()
    assert config.to_kwargs() == {}

    config.host = "127.0.0.1"
    config.port = 8080
    assert config.to_kwargs() == {"host": "127.0.0.1", "port": 8080}

    config.path = "/plex"
    kwargs = config.to_kwargs()
    assert kwargs["path"] == "/plex"


def test_find_records_handles_retrieve_error(monkeypatch):
    with _load_server(monkeypatch) as module:
        async def fail_retrieve(*args, **kwargs):
            raise RuntimeError("boom")

        async def fake_scroll(*args, **kwargs):
            payload = {"data": {"plex": {"rating_key": "1"}}}
            return ([types.SimpleNamespace(payload=payload)], None)

        monkeypatch.setattr(module.server.qdrant_client, "retrieve", fail_retrieve)
        monkeypatch.setattr(module.server.qdrant_client, "scroll", fake_scroll)

        records = asyncio.run(media_helpers._find_records(module.server, "123"))
        assert records and records[0].payload["data"]["plex"]["rating_key"] == "1"


def test_media_resources_cache_hits(monkeypatch):
    with _load_server(monkeypatch) as module:
        rating_key = "49915"
        poster_first = asyncio.run(module.media_poster.fn(identifier=rating_key))
        poster_cached = asyncio.run(module.media_poster.fn(identifier=rating_key))
        assert poster_cached == poster_first

        background_first = asyncio.run(module.media_background.fn(identifier=rating_key))
        background_cached = asyncio.run(module.media_background.fn(identifier=rating_key))
        assert background_cached == background_first


def test_rest_query_media_invalid_json(monkeypatch):
    with _load_server(monkeypatch) as module:
        client = TestClient(module.server.http_app())
        response = client.post(
            "/rest/query-media",
            content="not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 200


def test_rest_prompt_invalid_json(monkeypatch):
    with _load_server(monkeypatch) as module:
        prompt_cls = type(module.server._prompt_manager._prompts["media-info"])

        async def fake_render(self, arguments):
            assert arguments is None
            return [module.Message("ok")]

        monkeypatch.setattr(prompt_cls, "render", fake_render)
        client = TestClient(module.server.http_app())
        response = client.post(
            "/rest/prompt/media-info",
            content="not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 200


def test_rest_resource_content_types(monkeypatch):
    with _load_server(monkeypatch) as module:
        async def fake_read_resource(formatted: str):
            if formatted.endswith("binary"):
                return b"binary"
            if formatted.endswith("json"):
                return json.dumps({"value": 1})
            return "plain"

        monkeypatch.setattr(
            module.server._resource_manager,
            "read_resource",
            fake_read_resource,
        )

        client = TestClient(module.server.http_app())

        resp = client.get("/rest/resource/media-ids/binary")
        assert resp.content == b"binary"

        resp = client.get("/rest/resource/media-ids/json")
        assert resp.json()["value"] == 1

        resp = client.get("/rest/resource/media-ids/plain")
        assert resp.text == "plain"


def test_search_media_without_reranker(monkeypatch):
    with _load_server(monkeypatch) as module:
        payload = {
            "title": "Sample",
            "summary": "Summary",
            "plex": {"rating_key": "1", "title": "Sample"},
        }
        hits = [types.SimpleNamespace(payload=payload, score=0.2)]

        async def fake_query_points(*args, **kwargs):
            return types.SimpleNamespace(points=hits)

        async def immediate_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        monkeypatch.setattr(module.server.qdrant_client, "query_points", fake_query_points)
        monkeypatch.setattr(module.asyncio, "to_thread", immediate_to_thread)
        monkeypatch.setattr(module.server, "_reranker", None)
        monkeypatch.setattr(module.server, "_reranker_loaded", True)
        monkeypatch.setattr(module.server.settings, "use_reranker", False)

        results = asyncio.run(module.search_media.fn(query="test", limit=1))
        assert results[0]["plex"]["rating_key"] == "1"


def test_search_media_with_reranker(monkeypatch):
    with _load_server(monkeypatch) as module:
        payload_one = {
            "title": "First",
            "summary": "Summary",
            "plex": {
                "rating_key": "1",
                "title": "First",
                "summary": "First summary",
                "thumb": "thumb1",
                "art": "art1",
                "actors": [{"tag": "Actor Dict"}, "Actor String"],
            },
            "tmdb": {"overview": "Overview"},
            "directors": [{"tag": "Director"}],
            "writers": "Writer Name",
            "actors": [{"name": "Actor Dict"}, "Actor Text"],
            "tagline": ["Line one", "Line two"],
            "reviews": ["Great", ""],
        }
        payload_two = {
            "title": "Second",
            "summary": "Another",
            "plex": {"rating_key": "2", "title": "Second"},
        }
        hits = [
            types.SimpleNamespace(payload=payload_one, score=0.1),
            types.SimpleNamespace(payload=payload_two, score=0.2),
        ]

        async def fake_query_points(*args, **kwargs):
            return types.SimpleNamespace(points=list(hits))

        async def immediate_to_thread(fn, *args, **kwargs):
            return fn(*args, **kwargs)

        class DummyReranker:
            def predict(self, pairs):
                return [0.9, 0.1]

        monkeypatch.setattr(module.server.qdrant_client, "query_points", fake_query_points)
        monkeypatch.setattr(module.asyncio, "to_thread", immediate_to_thread)
        monkeypatch.setattr(module.server, "_reranker", DummyReranker())
        monkeypatch.setattr(module.server, "_reranker_loaded", True)
        monkeypatch.setattr(module.server.settings, "use_reranker", True)

        results = asyncio.run(module.search_media.fn(query="test", limit=2))
        assert [r["plex"]["rating_key"] for r in results] == ["1", "2"]


def test_query_media_filters(monkeypatch):
    with _load_server(monkeypatch) as module:
        captured: dict[str, object] = {}

        async def fake_query_points(*args, **kwargs):
            captured.update(kwargs)
            payload = {"title": "Result", "plex": {"rating_key": "1"}}
            return types.SimpleNamespace(points=[types.SimpleNamespace(payload=payload, score=1.0)])

        monkeypatch.setattr(module.server.qdrant_client, "query_points", fake_query_points)

        result = asyncio.run(
            module.query_media.fn(
                dense_query="dense",
                sparse_query="sparse",
                title="Title",
                type="movie",
                year=2024,
                year_from=2020,
                year_to=2025,
                added_after=10,
                added_before=20,
                actors="Actor",
                directors=["Director"],
                writers=("Writer",),
                genres=["Action"],
                collections=["Collection"],
                show_title="Show",
                season_number=1,
                episode_number=2,
                summary="summary",
                overview="overview",
                plot="plot",
                tagline="tagline",
                reviews="review",
                plex_rating_key="49915",
                imdb_id="tt1",
                tmdb_id=123,
                limit=2,
            )
        )

        assert result and result[0]["plex"]["rating_key"] == "1"
        query_filter = captured["query_filter"]
        assert query_filter is not None
        assert len(query_filter.must) >= 10
        assert isinstance(captured["query"], models.FusionQuery)
        prefetch = captured["prefetch"]
        assert prefetch is not None
        expected_prefetch_keys = {
            "type",
            "actors",
            "directors",
            "writers",
            "genres",
            "collections",
            "show_title",
            "data.plex.rating_key",
            "data.imdb.id",
        }
        for entry in prefetch:
            assert entry.filter is not None
            keys = {condition.key for condition in entry.filter.must}
            assert keys == expected_prefetch_keys


def test_query_media_filters_without_vectors(monkeypatch):
    with _load_server(monkeypatch) as module:
        captured: dict[str, object] = {}

        async def fake_query_points(*args, **kwargs):
            captured.update(kwargs)
            payload = {"title": "Result", "plex": {"rating_key": "1"}}
            return types.SimpleNamespace(
                points=[types.SimpleNamespace(payload=payload, score=1.0)]
            )

        monkeypatch.setattr(module.server.qdrant_client, "query_points", fake_query_points)

        result = asyncio.run(
            module.query_media.fn(
                type="movie",
                actors=["Actor"],
                limit=1,
            )
        )

        assert result and result[0]["plex"]["rating_key"] == "1"
        query_filter = captured["query_filter"]
        assert query_filter is not None
        keys = {condition.key for condition in query_filter.must}
        assert keys == {"type", "actors"}
        assert captured["prefetch"] is None


def test_openapi_schema_tool_without_params(monkeypatch):
    module = importlib.import_module("mcp_plex.server")

    @module.server.tool("coverage-noop")
    async def _coverage_noop() -> None:
        return None

    try:
        schema = module._build_openapi_schema()
        assert "/rest/coverage-noop" in schema["paths"]
    finally:
        module.server._tool_manager._tools.pop("coverage-noop", None)
        if hasattr(module, "_coverage_noop"):
            delattr(module, "_coverage_noop")
