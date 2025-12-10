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
from plexapi.exceptions import PlexApiException

from mcp_plex import loader
from mcp_plex import server as server_module
from mcp_plex.server import media as media_helpers
from mcp_plex.server.tools import media_library as media_library_tools
from pydantic import ValidationError


REMOVED_MEDIA_TOOL_NAMES = (
    "search_media",
    "recommend_media_like",
    "recommend_media",
)


def _reload_server_with_dummy_reranker(monkeypatch):
    monkeypatch.setenv("USE_RERANKER", "1")
    st_module = types.ModuleType("sentence_transformers")

    class Dummy:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    st_module.CrossEncoder = Dummy
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_module)
    module = importlib.reload(importlib.import_module("mcp_plex.server"))
    return module, Dummy


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
    module.server.settings.use_reranker = False
    try:
        yield module
    finally:
        asyncio.run(module.server.close())


def test_server_name_is_plex_media():
    module = importlib.reload(importlib.import_module("mcp_plex.server"))

    assert module.server.name == "Plex Media"


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
        media_items = asyncio.run(server.get_media.fn(identifier=movie_id))
        assert isinstance(media_items, list)
        assert media_items and media_items[0]["plex"]["rating_key"] == movie_id
        assert isinstance(media_items[0]["plex"]["added_at"], int)
        assert isinstance(media_items[0]["added_at"], int)

        imdb_items = asyncio.run(server.get_media.fn(identifier="tt8367814"))
        assert imdb_items and imdb_items[0]["plex"]["rating_key"] == movie_id

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
            server.server.cache.get_payload(movie_id)["plex"]["rating_key"] == movie_id
        )

        ids = json.loads(asyncio.run(server.media_ids.fn(identifier=movie_id)))
        assert ids["imdb"] == "tt8367814"

        structured = asyncio.run(
            server.query_media.fn(
                dense_query="crime comedy",
                title="Gentlemen",
                type="movie",
                directors=["Guy Ritchie"],
                limit=1,
            )
        )
        assert structured["results"][0]["identifiers"]["rating_key"] == movie_id
        assert "directors" in structured["results"][0]

        episode_structured = asyncio.run(
            server.query_media.fn(
                type="episode",
                show_title="Alien: Earth",
                season_number=1,
                episode_number=4,
                limit=1,
            )
        )
        episode_result = episode_structured["results"][0]
        assert episode_result["identifiers"]["rating_key"] == "61960"
        assert episode_result["show"] == "Alien: Earth"

        similar_structured = asyncio.run(
            server.query_media.fn(
                similar_to=["49915"],
                type="episode",
                limit=3,
            )
        )
        assert similar_structured["results"]
        similar_results = similar_structured["results"]
        assert {
            item["identifiers"]["rating_key"]
            for item in similar_results
            if isinstance(item.get("identifiers"), dict)
        } >= {"61960"}

        similar_structured_int = asyncio.run(
            server.query_media.fn(
                similar_to=49915,
                type="episode",
                limit=3,
            )
        )
        similar_results_int = similar_structured_int["results"]
        assert similar_results_int
        assert {
            item["identifiers"]["rating_key"]
            for item in similar_results_int
            if isinstance(item.get("identifiers"), dict)
        } >= {"61960"}

        empty_structured = asyncio.run(
            server.query_media.fn(
                similar_to="does-not-exist",
                type="movie",
                limit=1,
            )
        )
        assert empty_structured["results"] == []

        for removed_tool in REMOVED_MEDIA_TOOL_NAMES:
            assert not hasattr(server, removed_tool)

        with pytest.raises(ValueError):
            asyncio.run(server.media_item.fn(identifier="0"))
        with pytest.raises(ValueError):
            asyncio.run(server.media_ids.fn(identifier="0"))
        with pytest.raises(ValueError):
            asyncio.run(server.media_poster.fn(identifier="0"))
        with pytest.raises(ValueError):
            asyncio.run(server.media_background.fn(identifier="0"))


def test_get_watched_rating_keys_limits_history_items(monkeypatch):
    monkeypatch.setenv("PLEX_URL", "http://example.com")
    monkeypatch.setenv("PLEX_TOKEN", "token")
    monkeypatch.setenv("PLEX_RECOMMEND_USER", "history-user")
    monkeypatch.setenv("PLEX_RECOMMEND_HISTORY_LIMIT", "7")

    module = importlib.reload(importlib.import_module("mcp_plex.server"))

    assert module.server.settings.recommend_history_limit == 7

    history_calls: list[dict[str, Any]] = []

    class DummyHistoryItem:
        def __init__(self, rating_key: str) -> None:
            self.ratingKey = rating_key

    class DummyUser:
        def history(self, **kwargs):
            history_calls.append(kwargs)
            return [DummyHistoryItem(str(index)) for index in range(20)]

    class DummyAccount:
        def user(self, name: str):
            assert name == "history-user"
            return DummyUser()

    class DummyClient:
        def myPlexAccount(self):
            return DummyAccount()

    async def fake_get_client():
        return DummyClient()

    monkeypatch.setattr(module, "_get_plex_client", fake_get_client)

    watched = asyncio.run(module.server.get_watched_rating_keys())

    assert history_calls, "Expected history to be requested"
    assert history_calls[0].get("maxresults") == 7
    assert len(watched) == 7

    asyncio.run(module.server.close())


def test_media_library_tools_have_metadata(monkeypatch):
    module, _ = _reload_server_with_dummy_reranker(monkeypatch)
    try:
        expected = {
            "get_media": {
                "title": "Get media details",
                "description": (
                    "Retrieve media items by rating key, IMDb/TMDb ID or title."
                ),
                "operation": "lookup",
            },
            "query_media": {
                "title": "Query media library",
                "description": (
                    "Run a structured query against indexed payload fields and optional"
                    " vector searches."
                ),
                "operation": "query",
            },
            "new_movies": {
                "title": "Newest movies",
                "description": "Return the most recently added movies.",
                "operation": "recent-movies",
            },
            "new_shows": {
                "title": "Newest episodes",
                "description": "Return the most recently added TV episodes.",
                "operation": "recent-episodes",
            },
            "actor_movies": {
                "title": "Movies by actor",
                "description": (
                    "Return movies featuring the given actor, optionally filtered by"
                    " release year."
                ),
                "operation": "actor-filmography",
            },
        }

        for attr, details in expected.items():
            tool = getattr(module, attr)
            assert tool.title == details["title"]
            assert tool.description == details["description"]
            assert tool.meta is not None
            assert tool.meta.get("category") == "media-library"
            assert tool.meta.get("operation") == details["operation"]

        for attr in REMOVED_MEDIA_TOOL_NAMES:
            assert not hasattr(module, attr)
    finally:
        asyncio.run(module.server.close())


def test_query_media_year_range_schema_uses_integers(monkeypatch):
    module, _ = _reload_server_with_dummy_reranker(monkeypatch)
    try:
        properties = module.query_media.parameters["properties"]
        year_from_schema = properties["year_from"]
        year_to_schema = properties["year_to"]

        assert year_from_schema["type"] == "integer"
        assert year_from_schema.get("nullable") is True
        assert year_to_schema["type"] == "integer"
        assert year_to_schema.get("nullable") is True
    finally:
        asyncio.run(module.server.close())


def test_get_media_data_caches_external_ids(monkeypatch):
    with _load_server(monkeypatch) as server:
        call_count = 0

        original_find_records = media_helpers._find_records

        async def _counting_find_records(plex_server, identifier: str, limit: int = 1):
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


def test_get_media_data_ignores_mismatched_cached_identifier(monkeypatch):
    with _load_server(monkeypatch) as server:
        plex_server = server.server

        call_count = 0

        original_find_records = media_helpers._find_records

        async def _counting_find_records(plex_server, identifier: str, limit: int = 1):
            nonlocal call_count
            call_count += 1
            return await original_find_records(plex_server, identifier, limit=limit)

        monkeypatch.setattr(media_helpers, "_find_records", _counting_find_records)

        server.server.cache.set_payload(
            "522627",
            {
                "plex": {"rating_key": "12345", "title": "Fake Movie"},
            },
        )

        data = asyncio.run(media_helpers._get_media_data(plex_server, "522627"))

        assert data["plex"]["rating_key"] == "49915"
        assert call_count == 1


def test_new_media_tools(monkeypatch):
    with _load_server(monkeypatch) as server:
        movies = asyncio.run(server.new_movies.fn(limit=1))
        assert movies["results"]
        movie = movies["results"][0]
        assert movie["identifiers"]["rating_key"] == "49915"

        shows = asyncio.run(server.new_shows.fn(limit=1))
        assert shows["results"]
        episode = shows["results"][0]
        assert episode["identifiers"]["rating_key"] == "61960"
        assert episode["show"] == "Alien: Earth"
        assert episode["season"] == 1
        assert episode["episode"] == 4


def test_actor_movies(monkeypatch):
    with _load_server(monkeypatch) as server:
        movies = asyncio.run(
            server.actor_movies.fn(
                actor="Matthew McConaughey",
                limit=1,
            )
        )
        assert movies["results"][0]["title"] == "The Gentlemen"

        none = asyncio.run(
            server.actor_movies.fn(
                actor="Matthew McConaughey",
                year_from=1990,
                year_to=1999,
            )
        )
        assert none["results"] == []


def test_play_media_requires_configuration(monkeypatch):
    with _load_server(monkeypatch) as server:
        with pytest.raises(RuntimeError):
            asyncio.run(server.play_media.fn(identifier="49915", player="Living Room"))


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


def test_queue_media_adds_to_queue(monkeypatch):
    monkeypatch.setenv("PLEX_URL", "http://plex.test:32400")
    monkeypatch.setenv("PLEX_TOKEN", "token")
    monkeypatch.setenv(
        "PLEX_PLAYER_ALIASES",
        json.dumps(
            {
                "machine-123": ["Living Room"],
                "client-abc": ["Living Room"],
                "machine-123:client-abc": ["Living Room"],
            }
        ),
    )

    class FakeTimeline:
        def __init__(self) -> None:
            self.playQueueID = 42
            self.playQueueItemID = 101
            self.playQueueVersion = 3
            self.state = "playing"
            self.type = "video"

    class FakeClient:
        def __init__(self) -> None:
            self.machineIdentifier = "machine-123"
            self.clientIdentifier = "client-abc"
            self.provides = "player,controller"
            self.address = "10.0.0.5"
            self.port = 32500
            self.product = "Plex for Apple TV"
            self.title = "Plex for Apple TV"
            self._timeline = FakeTimeline()

        def timelines(self, wait: int = 0):
            return [self._timeline]

        @property
        def timeline(self):
            return self._timeline

    class FakeMedia:
        def __init__(self, key: str) -> None:
            self.key = key

    class FakePlex:
        def __init__(self, baseurl: str, token: str) -> None:
            assert baseurl.rstrip("/") == "http://plex.test:32400"
            assert token == "token"
            self.machineIdentifier = "server-001"
            self._client = FakeClient()
            self.fetch_requests: list[str] = []

        def clients(self) -> list[FakeClient]:
            return [self._client]

        def fetchItem(self, key: str) -> FakeMedia:
            self.fetch_requests.append(key)
            return FakeMedia(key)

    queue_calls: list[dict[str, object]] = []
    get_calls: list[tuple[int, dict[str, object]]] = []

    class FakePlayQueue:
        def __init__(self) -> None:
            self.playQueueTotalCount = 1
            self.playQueueVersion = 5

        def addItem(self, item: FakeMedia, playNext: bool = False, refresh: bool = True):
            queue_calls.append(
                {
                    "item": item,
                    "playNext": playNext,
                    "refresh": refresh,
                }
            )
            self.playQueueTotalCount += 1
            self.playQueueVersion += 1
            return self

    fake_queue = FakePlayQueue()

    class _FakePlayQueue:
        @classmethod
        def get(cls, plex_server, playQueueID: int, **kwargs):  # type: ignore[no-untyped-def]
            get_calls.append((playQueueID, kwargs))
            return fake_queue

    with _load_server(monkeypatch) as server:
        monkeypatch.setattr(server, "PlexServerClient", FakePlex)
        monkeypatch.setattr(server, "PlayQueue", _FakePlayQueue, raising=False)

        result_next = asyncio.run(
            server.queue_media.fn(identifier="49915", player="Living Room", play_next=True)
        )

        assert result_next["player"] == "Living Room"
        assert result_next["rating_key"] == "49915"
        assert result_next["position"] == "next"
        assert result_next["queue_size"] == fake_queue.playQueueTotalCount
        assert queue_calls[-1]["playNext"] is True
        assert queue_calls[-1]["refresh"] is True
        assert server.server._plex_client.fetch_requests == ["/library/metadata/49915"]
        assert get_calls[-1][0] == 42
        assert get_calls[-1][1]["own"] is True

        queue_calls.clear()
        fake_queue.playQueueTotalCount = 3
        fake_queue.playQueueVersion = 7
        server.server._plex_client.fetch_requests.clear()

        result_end = asyncio.run(
            server.queue_media.fn(identifier="49915", player="Living Room", play_next=False)
        )

        assert result_end["player"] == "Living Room"
        assert result_end["rating_key"] == "49915"
        assert result_end["position"] == "end"
        assert result_end["queue_size"] == fake_queue.playQueueTotalCount
        assert queue_calls[-1]["playNext"] is False
        assert server.server._plex_client.fetch_requests == ["/library/metadata/49915"]

def test_play_media_allows_controller_only_client(monkeypatch):
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
            self.play_requests: list[dict[str, Any]] = []

        def playMedia(self, *args: Any, **kwargs: Any) -> None:
            self.play_requests.append({"args": args, "kwargs": kwargs})

    class FakePlex:
        def __init__(self, baseurl: str, token: str) -> None:
            assert baseurl.rstrip("/") == "http://plex.test:32400"
            assert token == "token"
            self.machineIdentifier = "server-001"
            self._client = FakeClient()
            self.fetch_requests: list[str] = []

        def clients(self) -> list[FakeClient]:
            return [self._client]

        def fetchItem(self, key: str) -> Any:
            self.fetch_requests.append(key)
            return types.SimpleNamespace(key=key)

    with _load_server(monkeypatch) as server:
        fake_plex = FakePlex("http://plex.test:32400", "token")
        monkeypatch.setattr(
            server,
            "PlexServerClient",
            lambda baseurl, token: fake_plex,
        )
        server._plex_client = None
        server._plex_identity = None
        result = asyncio.run(
            server.play_media.fn(identifier="49915", player="machine-999")
        )

        assert result["player"] == "Controller Only"
        assert result["rating_key"] == "49915"
        assert result["player_capabilities"] == ["controller"]
        assert fake_plex.fetch_requests == ["/library/metadata/49915"]
        assert len(fake_plex._client.play_requests) == 1
        play_call = fake_plex._client.play_requests[0]
        assert play_call["kwargs"]["machineIdentifier"] == "server-001"
        assert play_call["kwargs"]["offset"] == 0


def test_playback_control_tools_send_commands(monkeypatch):
    monkeypatch.setenv("PLEX_URL", "http://plex.test:32400")
    monkeypatch.setenv("PLEX_TOKEN", "token")

    class FakeAudioStream:
        def __init__(self, stream_id: int, language: str, channels: int) -> None:
            self.id = stream_id
            self.language = language
            self.channels = channels

    class FakeSubtitleStream:
        def __init__(
            self,
            stream_id: int,
            language: str,
            *,
            forced: bool = False,
            default: bool = False,
        ) -> None:
            self.id = stream_id
            self.language = language
            self.forced = forced
            self.default = default
            self.streamType = 3

    class FakeSessionPlayer:
        def __init__(self, machine_identifier: str, client_identifier: str) -> None:
            self.machineIdentifier = machine_identifier
            self.clientIdentifier = client_identifier

    class FakeSession:
        def __init__(
            self,
            players: list[FakeSessionPlayer],
            audio_streams: list[FakeAudioStream],
            subtitle_streams: list[FakeSubtitleStream],
        ) -> None:
            self.players = players
            self._audio_streams = audio_streams
            self._subtitle_streams = subtitle_streams

        def audioStreams(self) -> list[FakeAudioStream]:
            return list(self._audio_streams)

        def subtitleStreams(self) -> list[FakeSubtitleStream]:
            return list(self._subtitle_streams)

    class FakeClient:
        def __init__(self) -> None:
            self.machineIdentifier = "machine-123"
            self.clientIdentifier = "client-abc"
            self.provides = "player,controller"
            self.address = "10.0.0.5"
            self.port = 32500
            self.product = "Plex for Apple TV"
            self.title = "Plex for Apple TV"
            self.pause_calls: list[str] = []
            self.play_calls: list[str] = []
            self.next_calls: list[str] = []
            self.previous_calls: list[str] = []
            self.forward_calls: list[str] = []
            self.back_calls: list[str] = []
            self.subtitle_calls: list[dict[str, Any]] = []
            self.audio_calls: list[dict[str, Any]] = []

        def pause(self, mtype: str = "video") -> None:
            self.pause_calls.append(mtype)

        def play(self, mtype: str = "video") -> None:
            self.play_calls.append(mtype)

        def skipNext(self, mtype: str = "video") -> None:
            self.next_calls.append(mtype)

        def skipPrevious(self, mtype: str = "video") -> None:
            self.previous_calls.append(mtype)

        def stepForward(self, mtype: str = "video") -> None:
            self.forward_calls.append(mtype)

        def stepBack(self, mtype: str = "video") -> None:
            self.back_calls.append(mtype)

        def setSubtitleStream(self, stream: Any, mtype: str = "video") -> None:
            self.subtitle_calls.append({"stream": stream, "media_type": mtype})

        def setAudioStream(self, stream: FakeAudioStream, mtype: str = "video") -> None:
            self.audio_calls.append({"stream": stream, "media_type": mtype})

    class FakePlex:
        def __init__(self, baseurl: str, token: str) -> None:
            assert baseurl.rstrip("/") == "http://plex.test:32400"
            assert token == "token"
            self.machineIdentifier = "server-001"
            self._client = FakeClient()
            self._sessions = [
                FakeSession(
                    [
                        FakeSessionPlayer(
                            self._client.machineIdentifier,
                            self._client.clientIdentifier,
                        )
                    ],
                    [
                        FakeAudioStream(101, "eng", 2),
                        FakeAudioStream(202, "eng", 6),
                        FakeAudioStream(303, "spa", 6),
                    ],
                    [
                        FakeSubtitleStream(401, "eng"),
                        FakeSubtitleStream(505, "spa"),
                    ],
                )
            ]

        def clients(self) -> list[FakeClient]:
            return [self._client]

        def sessions(self) -> list[FakeSession]:
            return self._sessions

        def fetchItem(self, key: str) -> Any:
            return types.SimpleNamespace(key=key)

    with _load_server(monkeypatch) as server:
        fake_plex = FakePlex("http://plex.test:32400", "token")
        monkeypatch.setattr(
            server,
            "PlexServerClient",
            lambda baseurl, token: fake_plex,
        )
        server._plex_client = None
        server._plex_identity = None

        pause_result = asyncio.run(server.pause_media.fn(player="Plex for Apple TV"))
        assert pause_result == {
            "player": "Plex for Apple TV",
            "command": "pause",
            "media_type": "video",
            "player_capabilities": ["controller", "player"],
            "success": True,
        }
        assert fake_plex._client.pause_calls == ["video"]

        resume_result = asyncio.run(server.resume_media.fn(player="Plex for Apple TV"))
        assert resume_result["command"] == "resume"
        assert resume_result["player"] == "Plex for Apple TV"
        assert resume_result["success"] is True
        assert fake_plex._client.play_calls == ["video"]

        next_result = asyncio.run(server.next_media.fn(player="Plex for Apple TV"))
        assert next_result["command"] == "next"
        assert next_result["success"] is True
        assert fake_plex._client.next_calls == ["video"]

        previous_result = asyncio.run(
            server.previous_media.fn(player="Plex for Apple TV")
        )
        assert previous_result["command"] == "previous"
        assert previous_result["success"] is True
        assert fake_plex._client.previous_calls == ["video"]

        fastforward_result = asyncio.run(
            server.fastforward_media.fn(player="Plex for Apple TV", media_type="music")
        )
        assert fastforward_result["command"] == "fastforward"
        assert fastforward_result["media_type"] == "music"
        assert fastforward_result["success"] is True
        assert fake_plex._client.forward_calls == ["music"]

        rewind_result = asyncio.run(server.rewind_media.fn(player="Plex for Apple TV"))
        assert rewind_result["command"] == "rewind"
        assert rewind_result["success"] is True
        assert fake_plex._client.back_calls == ["video"]

        subtitle_result = asyncio.run(
            server.set_subtitle.fn(
                player="Plex for Apple TV",
                subtitle_language="spa",
                media_type="video",
            )
        )
        assert subtitle_result == {
            "player": "Plex for Apple TV",
            "command": "set-subtitle",
            "media_type": "video",
            "subtitle_language": "spa",
            "subtitle_stream_id": 505,
            "player_capabilities": ["controller", "player"],
            "success": True,
        }
        assert len(fake_plex._client.subtitle_calls) == 1
        subtitle_call = fake_plex._client.subtitle_calls[0]
        subtitle_stream = subtitle_call["stream"]
        assert isinstance(subtitle_stream, FakeSubtitleStream)
        assert subtitle_stream.id == 505
        assert subtitle_stream.language == "spa"
        assert subtitle_call["media_type"] == "video"

        audio_result = asyncio.run(
            server.set_audio.fn(
                player="Plex for Apple TV",
                audio_language="eng",
                media_type="video",
            )
        )
        assert audio_result["player"] == "Plex for Apple TV"
        assert audio_result["command"] == "set-audio"
        assert audio_result["media_type"] == "video"
        assert audio_result["audio_language"] == "eng"
        assert audio_result["player_capabilities"] == ["controller", "player"]
        assert audio_result["audio_channels"] == 6
        assert audio_result["audio_stream_id"] == 202
        assert audio_result["success"] is True
        assert len(fake_plex._client.audio_calls) == 1
        audio_call = fake_plex._client.audio_calls[0]
        stream = audio_call["stream"]
        assert isinstance(stream, FakeAudioStream)
        assert stream.id == 202
        assert stream.language == "eng"
        assert stream.channels == 6
        assert audio_call["media_type"] == "video"


def test_playback_control_reports_command_errors(monkeypatch):
    monkeypatch.setenv("PLEX_URL", "http://plex.test:32400")
    monkeypatch.setenv("PLEX_TOKEN", "token")

    class FaultyClient:
        def __init__(self) -> None:
            self.machineIdentifier = "client-001"
            self.clientIdentifier = "client-001"
            self.title = "Plex for Apple TV"
            self.provides = {"controller", "player"}
            self.address = "127.0.0.1"
            self.port = 32500

        def pause(self, mtype: str = "video") -> None:  # noqa: ARG002 - plexapi parity
            raise PlexApiException("Mock pause failure")

    class FaultyPlex:
        def __init__(self, baseurl: str, token: str) -> None:
            assert baseurl.rstrip("/") == "http://plex.test:32400"
            assert token == "token"
            self._client = FaultyClient()

        def clients(self) -> list[FaultyClient]:
            return [self._client]

        def sessions(self) -> list[Any]:
            return []

    with _load_server(monkeypatch) as server:
        faulty_plex = FaultyPlex("http://plex.test:32400", "token")
        monkeypatch.setattr(
            server,
            "PlexServerClient",
            lambda baseurl, token: faulty_plex,
        )
        server._plex_client = None
        server._plex_identity = None

        result = asyncio.run(server.pause_media.fn(player="Plex for Apple TV"))
        assert result == {
            "player": "Plex for Apple TV",
            "command": "pause",
            "media_type": "video",
            "player_capabilities": ["controller", "player"],
            "success": False,
            "error": "Mock pause failure",
        }


def test_set_subtitle_requires_language(monkeypatch):
    monkeypatch.setenv("PLEX_URL", "http://plex.test:32400")
    monkeypatch.setenv("PLEX_TOKEN", "token")

    class FakeClient:
        def __init__(self) -> None:
            self.machineIdentifier = "machine-123"
            self.clientIdentifier = "client-abc"
            self.provides = "player"
            self.title = "Living Room"

        def setSubtitleStream(self, language: str, mtype: str = "video") -> None:  # noqa: ARG002
            raise AssertionError("Should not be called")

    class FakePlex:
        def __init__(self, baseurl: str, token: str) -> None:
            assert baseurl.rstrip("/") == "http://plex.test:32400"
            assert token == "token"
            self.machineIdentifier = "server-001"
            self._client = FakeClient()

        def clients(self) -> list[FakeClient]:
            return [self._client]

    with _load_server(monkeypatch) as server:
        fake_plex = FakePlex("http://plex.test:32400", "token")
        monkeypatch.setattr(
            server,
            "PlexServerClient",
            lambda baseurl, token: fake_plex,
        )
        server._plex_client = None
        server._plex_identity = None

        result = asyncio.run(
            server.set_subtitle.fn(player="Living Room", subtitle_language="")
        )
        assert result["success"] is False
        assert "subtitle language" in result["error"]
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


def test_match_player_alias_name_maps_to_identifier(monkeypatch):
    monkeypatch.setattr(
        server_module.server,
        "_settings",
        server_module.Settings.model_validate(
            {"PLEX_PLAYER_ALIASES": {"Movie Room": "6B4C9A5E-E333-4DB3-A8E7-49C8F5933EB1"}}
        ),
    )

    players: list[server_module.PlexPlayerMetadata] = [
        {
            "display_name": "Plex for Apple TV",
            "name": "Plex for Apple TV",
            "product": "Apple TV",
            "machine_identifier": "6B4C9A5E-E333-4DB3-A8E7-49C8F5933EB1",
            "client_identifier": "client-1",
            "friendly_names": [],
            "provides": {"player"},
            "client": None,
        }
    ]

    matched = server_module._match_player("Movie Room", players)

    assert matched is players[0]


def test_match_player_alias_fuzzy_identifier(monkeypatch):
    original_settings = server_module.server.settings
    fuzzy_settings = original_settings.model_copy(
        update={"plex_player_aliases": {"Movie Room": ("Machine 1",)}}
    )

    monkeypatch.setattr(server_module.server, "_settings", fuzzy_settings)

    try:
        players: list[server_module.PlexPlayerMetadata] = [
            {
                "display_name": "Plex for Apple TV",
                "name": "Plex for Apple TV",
                "product": "Apple TV",
                "machine_identifier": "machine-1",
                "client_identifier": "client-1",
                "friendly_names": [],
                "provides": {"player"},
                "client": None,
            }
        ]

        matched = server_module._match_player("Movie Room", players)
    finally:
        monkeypatch.setattr(server_module.server, "_settings", original_settings)

    assert matched is players[0]


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
        "Failed to import CrossEncoder" in message for message in caplog.messages
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


def test_ensure_reranker_uses_thread_executor(monkeypatch):
    monkeypatch.setenv(
        "RERANKER_MODEL",
        "sentence-transformers/test-cross-encoder",
    )
    module, Dummy = _reload_server_with_dummy_reranker(monkeypatch)
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []

    async def fake_to_thread(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((fn, args, kwargs))
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def exercise():
        reranker = await module.server.ensure_reranker()
        assert isinstance(reranker, Dummy)
        assert reranker is await module.server.ensure_reranker()

    asyncio.run(exercise())

    assert len(calls) == 1
    fn, args, _ = calls[0]
    assert fn is Dummy
    assert args == ("sentence-transformers/test-cross-encoder",)
    asyncio.run(module.server.close())


def test_ensure_reranker_concurrent_calls_share_single_instance(monkeypatch):
    module, Dummy = _reload_server_with_dummy_reranker(monkeypatch)
    call_count = 0

    async def fake_to_thread(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    async def exercise():
        results = await asyncio.gather(
            *(module.server.ensure_reranker() for _ in range(5))
        )
        assert call_count == 1
        assert all(reranker is results[0] for reranker in results)

    asyncio.run(exercise())
    asyncio.run(module.server.close())


def test_rest_endpoints(monkeypatch):
    with _load_server(monkeypatch) as module:
        client = TestClient(module.server.http_app())

        resp = client.post("/rest/get-media", json={"identifier": "49915"})
        assert resp.status_code == 200
        raw_payload = resp.json()
        assert isinstance(raw_payload, list)
        assert raw_payload[0]["plex"]["rating_key"] == "49915"

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
        assert get_media_schema["properties"]["identifier"]["description"].startswith(
            "Rating key"
        )
        assert "summarize_for_llm" not in get_media_schema["properties"]

        assert "/rest/search-media" not in spec["paths"]
        assert "/rest/recommend-media" not in spec["paths"]
        assert "/rest/recommend-media-like" not in spec["paths"]
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

        background_first = asyncio.run(
            module.media_background.fn(identifier=rating_key)
        )
        background_cached = asyncio.run(
            module.media_background.fn(identifier=rating_key)
        )
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



def test_query_media_filters(monkeypatch):
    with _load_server(monkeypatch) as module:
        captured: dict[str, object] = {}

        async def fake_query_points(*args, **kwargs):
            captured.update(kwargs)
            payload = {"title": "Result", "plex": {"rating_key": "1"}}
            return types.SimpleNamespace(
                points=[types.SimpleNamespace(payload=payload, score=1.0)]
            )

        monkeypatch.setattr(
            module.server.qdrant_client, "query_points", fake_query_points
        )

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

        result_entry = result["results"][0]
        assert result_entry["identifiers"]["rating_key"] == "1"
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

        monkeypatch.setattr(
            module.server.qdrant_client, "query_points", fake_query_points
        )

        result = asyncio.run(
            module.query_media.fn(
                type="movie",
                actors=["Actor"],
                limit=1,
            )
        )

        assert result["results"][0]["identifiers"]["rating_key"] == "1"
        query_filter = captured["query_filter"]
        assert query_filter is not None
        keys = {condition.key for condition in query_filter.must}
        assert keys == {"type", "actors"}
        assert captured["prefetch"] is None


def test_query_media_infers_show_title_for_episode_requests(monkeypatch):
    with _load_server(monkeypatch) as module:
        captured: dict[str, object] = {}

        async def fake_find_records(server, identifier, limit=5):
            payload = {
                "show_title": "South Park",
                "data": {
                    "plex": {
                        "rating_key": "show",
                        "grandparent_title": "South Park",
                    }
                },
            }
            return [types.SimpleNamespace(payload=payload, id="show")]

        async def fake_query_points(*args, **kwargs):
            captured.update(kwargs)
            payload = {"title": "Result", "plex": {"rating_key": "1"}}
            return types.SimpleNamespace(
                points=[types.SimpleNamespace(payload=payload, score=1.0)]
            )

        monkeypatch.setattr(media_helpers, "_find_records", fake_find_records)
        monkeypatch.setattr(
            module.server.qdrant_client, "query_points", fake_query_points
        )

        result = asyncio.run(
            module.query_media.fn(
                title="south park",
                season_number=23,
                episode_number=7,
                limit=1,
            )
        )

        assert result["results"], "expected at least one result"
        query_filter = captured["query_filter"]
        assert query_filter is not None
        keys = {condition.key for condition in query_filter.must}
        assert "show_title" in keys
        assert "type" in keys
        for condition in query_filter.must:
            if condition.key == "show_title":
                assert condition.match.value == "South Park"


def test_query_media_applies_reranker_when_available(monkeypatch):
    with _load_server(monkeypatch) as module:
        module.server.settings.use_reranker = True
        calls: list[list[tuple[str, str]]] = []

        class DummyReranker:
            def predict(self, pairs):
                calls.append(pairs)
                return [0.1, 0.9]

        async def fake_ensure_reranker():
            return DummyReranker()

        async def fake_to_thread(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
            return fn(*args, **kwargs)

        async def fake_query_points(*args, **kwargs):
            payload_first = {"title": "First", "summary": "Alpha", "plex": {"rating_key": "1"}}
            payload_second = {"title": "Second", "summary": "Beta", "plex": {"rating_key": "2"}}
            return types.SimpleNamespace(
                points=[
                    types.SimpleNamespace(payload=payload_first, score=0.7),
                    types.SimpleNamespace(payload=payload_second, score=0.6),
                ]
            )

        monkeypatch.setattr(module.server, "ensure_reranker", fake_ensure_reranker)
        monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
        monkeypatch.setattr(
            module.server.qdrant_client, "query_points", fake_query_points
        )

        result = asyncio.run(
            module.query_media.fn(dense_query="buddy comedy", limit=2)
        )

        identifiers = [
            item["identifiers"]["rating_key"]
            for item in result["results"]
            if isinstance(item.get("identifiers"), dict)
        ]
        assert identifiers == ["2", "1"], "expected reranker to reorder results"
        assert calls, "expected reranker to receive scoring pairs"
        assert calls[0][0][0] == "buddy comedy"


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
