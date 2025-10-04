import asyncio
import builtins
import importlib
import io
import json
import sys
import types
from datetime import datetime
from pathlib import Path

import httpx
from qdrant_client import models
import pytest

from mcp_plex import loader
from mcp_plex.imdb_cache import IMDbCache
from mcp_plex.loader import (
    _build_plex_item,
    _extract_external_ids,
    _fetch_imdb,
    _fetch_imdb_batch,
    _fetch_tmdb_episode,
    _fetch_tmdb_movie,
    _fetch_tmdb_show,
    _load_from_sample,
    _load_imdb_retry_queue,
    _persist_imdb_retry_queue,
    _process_imdb_retry_queue,
    _resolve_dense_model_params,
    resolve_tmdb_season_number,
)
from mcp_plex.types import (
    AggregatedItem,
    IMDbName,
    IMDbRating,
    IMDbTitle,
    PlexGuid,
    PlexItem,
    PlexPerson,
    TMDBMovie,
    TMDBSeason,
    TMDBShow,
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


def test_extract_external_ids():
    guid_objs = [
        types.SimpleNamespace(id="imdb://tt0133093"),
        types.SimpleNamespace(id="tmdb://603"),
    ]
    item = types.SimpleNamespace(guids=guid_objs)
    ids = _extract_external_ids(item)
    assert ids.imdb == "tt0133093"
    assert ids.tmdb == "603"


def test_extract_external_ids_missing_values():
    item = types.SimpleNamespace(guids=None)
    ids = _extract_external_ids(item)
    assert ids.imdb is None
    assert ids.tmdb is None


def test_load_from_sample_returns_items():
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    items = _load_from_sample(sample_dir)
    assert len(items) == 2
    assert {i.plex.type for i in items} == {"movie", "episode"}


def test_build_plex_item_handles_full_metadata():
    guid_objs = [
        types.SimpleNamespace(id="imdb://tt0133093"),
        types.SimpleNamespace(id="tmdb://603"),
    ]
    raw = types.SimpleNamespace(
        ratingKey="603",
        guid="plex://movie/603",
        type="movie",
        title="The Matrix",
        summary="A hacker discovers the nature of his reality.",
        year=1999,
        guids=guid_objs,
        thumb="matrix.jpg",
        art="matrix_art.jpg",
        tagline="Welcome to the Real World",
        contentRating="R",
        directors=[types.SimpleNamespace(id=1, tag="Lana Wachowski", thumb="lana.jpg")],
        writers=[types.SimpleNamespace(id=2, tag="Lilly Wachowski", thumb="lilly.jpg")],
        actors=[types.SimpleNamespace(id=3, tag="Keanu Reeves", thumb="neo.jpg", role="Neo")],
    )

    item = _build_plex_item(raw)
    assert item.rating_key == "603"
    assert item.directors[0].tag == "Lana Wachowski"
    assert item.actors[0].role == "Neo"


def test_build_plex_item_missing_metadata_defaults():
    raw = types.SimpleNamespace(ratingKey="1", guid="g", type="movie", title="T")
    item = _build_plex_item(raw)
    assert item.directors == []
    assert item.writers == []
    assert item.actors == []


def test_fetch_functions_success_and_failure():
    async def imdb_mock(request):
        if "good" in str(request.url):
            return httpx.Response(200, json={"id": "tt1", "type": "movie", "primaryTitle": "T"})
        return httpx.Response(404)

    async def tmdb_movie_mock(request):
        assert request.headers.get("Authorization") == "Bearer k"
        if "good" in str(request.url):
            return httpx.Response(200, json={"id": 1, "title": "M"})
        return httpx.Response(404)

    async def tmdb_show_mock(request):
        assert request.headers.get("Authorization") == "Bearer k"
        if "good" in str(request.url):
            return httpx.Response(200, json={"id": 1, "name": "S"})
        return httpx.Response(404)

    async def tmdb_episode_mock(request):
        assert request.headers.get("Authorization") == "Bearer k"
        if "/tv/1/season/2/episode/3" in str(request.url):
            return httpx.Response(200, json={"id": 1, "name": "E"})
        return httpx.Response(404)

    async def main():
        imdb_transport = httpx.MockTransport(imdb_mock)
        movie_transport = httpx.MockTransport(tmdb_movie_mock)
        show_transport = httpx.MockTransport(tmdb_show_mock)
        episode_transport = httpx.MockTransport(tmdb_episode_mock)

        async with httpx.AsyncClient(transport=imdb_transport) as client:
            assert (await _fetch_imdb(client, "good")) is not None
            assert (await _fetch_imdb(client, "bad")) is None

        async with httpx.AsyncClient(transport=movie_transport) as client:
            assert (await _fetch_tmdb_movie(client, "good", "k")) is not None
            assert (await _fetch_tmdb_movie(client, "bad", "k")) is None

        async with httpx.AsyncClient(transport=show_transport) as client:
            assert (await _fetch_tmdb_show(client, "good", "k")) is not None
            assert (await _fetch_tmdb_show(client, "bad", "k")) is None

        async with httpx.AsyncClient(transport=episode_transport) as client:
            assert (await _fetch_tmdb_episode(client, 1, 2, 3, "k")) is not None
            assert (await _fetch_tmdb_episode(client, 1, 2, 4, "k")) is None

    asyncio.run(main())


def test_fetch_functions_handle_http_error():
    def raise_error(request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        raise httpx.ConnectError("boom", request=request)

    async def main() -> None:
        transport = httpx.MockTransport(raise_error)
        async with httpx.AsyncClient(transport=transport) as client:
            assert await _fetch_imdb(client, "tt1") is None
        async with httpx.AsyncClient(transport=transport) as client:
            assert await _fetch_tmdb_movie(client, "1", "k") is None
        async with httpx.AsyncClient(transport=transport) as client:
            assert await _fetch_tmdb_show(client, "1", "k") is None
        async with httpx.AsyncClient(transport=transport) as client:
            assert await _fetch_tmdb_episode(client, 1, 1, 1, "k") is None

    asyncio.run(main())


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


def test_fetch_imdb_batch(tmp_path, monkeypatch):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(loader, "_imdb_cache", IMDbCache(cache_path))

    async def imdb_mock(request):
        params = request.url.params
        assert sorted(params.get_list("titleIds")) == ["tt1", "tt2"]
        return httpx.Response(
            200,
            json={
                "titles": [
                    {"id": "tt1", "type": "movie", "primaryTitle": "A"},
                    {"id": "tt2", "type": "movie", "primaryTitle": "B"},
                ]
            },
        )

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(imdb_mock)) as client:
            result = await _fetch_imdb_batch(client, ["tt1", "tt2"])
            assert result["tt1"] and result["tt1"].primaryTitle == "A"
            assert result["tt2"] and result["tt2"].primaryTitle == "B"

    asyncio.run(main())
    data = json.loads(cache_path.read_text())
    assert set(data.keys()) == {"tt1", "tt2"}


def test_fetch_imdb_batch_chunks(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.json"
    monkeypatch.setattr(loader, "_imdb_cache", IMDbCache(cache_path))

    calls: list[list[str]] = []

    async def imdb_mock(request):
        ids = request.url.params.get_list("titleIds")
        calls.append(ids)
        return httpx.Response(
            200,
            json={
                "titles": [
                    {"id": i, "type": "movie", "primaryTitle": i} for i in ids
                ]
            },
        )

    async def main():
        ids = [f"tt{i}" for i in range(6)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(imdb_mock)) as client:
            result = await _fetch_imdb_batch(client, ids)
            assert set(result.keys()) == set(ids)

    asyncio.run(main())
    assert len(calls) == 2
    assert all(len(c) <= 5 for c in calls)


def test_fetch_imdb_batch_all_cached(monkeypatch, tmp_path):
    cache_path = tmp_path / "cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "tt0111161": {
                    "id": "tt0111161",
                    "type": "movie",
                    "primaryTitle": "The Shawshank Redemption",
                },
                "tt0068646": {
                    "id": "tt0068646",
                    "type": "movie",
                    "primaryTitle": "The Godfather",
                },
            }
        )
    )
    monkeypatch.setattr(loader, "_imdb_cache", IMDbCache(cache_path))

    async def error_mock(request):
        raise AssertionError("network should not be called")

    async def main():
        async with httpx.AsyncClient(transport=httpx.MockTransport(error_mock)) as client:
            result = await _fetch_imdb_batch(client, ["tt0111161", "tt0068646"])
            assert result["tt0111161"].primaryTitle == "The Shawshank Redemption"
            assert result["tt0068646"].primaryTitle == "The Godfather"

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


def test_resolve_tmdb_season_number_matches_name():
    episode = types.SimpleNamespace(parentIndex=2018, parentTitle="2018")
    show = TMDBShow(
        id=1,
        name="Show",
        seasons=[TMDBSeason(season_number=14, name="2018")],
    )
    assert resolve_tmdb_season_number(show, episode) == 14


def test_resolve_tmdb_season_number_matches_air_date():
    episode = types.SimpleNamespace(parentIndex=2018, parentTitle="Season 2018")
    show = TMDBShow(
        id=1,
        name="Show",
        seasons=[TMDBSeason(season_number=16, name="Season 16", air_date="2018-01-03")],
    )
    assert resolve_tmdb_season_number(show, episode) == 16


def test_resolve_tmdb_season_number_parent_year_fallback():
    episode = types.SimpleNamespace(
        parentIndex="Special",
        parentTitle="Special",
        parentYear=2018,
    )
    show = TMDBShow(
        id=1,
        name="Show",
        seasons=[TMDBSeason(season_number=5, name="Season 5", air_date="2018-06-01")],
    )
    assert resolve_tmdb_season_number(show, episode) == 5


def test_resolve_tmdb_season_number_numeric_match():
    episode = types.SimpleNamespace(parentIndex=2, parentTitle="Season 2")
    show = TMDBShow(
        id=1,
        name="Show",
        seasons=[TMDBSeason(season_number=2, name="Season 2")],
    )
    assert resolve_tmdb_season_number(show, episode) == 2


def test_resolve_tmdb_season_number_title_year():
    episode = types.SimpleNamespace(parentTitle="2018")
    show = TMDBShow(
        id=1,
        name="Show",
        seasons=[TMDBSeason(season_number=7, name="Season 7", air_date="2018-02-03")],
    )
    assert resolve_tmdb_season_number(show, episode) == 7


def test_resolve_tmdb_season_number_parent_index_str():
    episode = types.SimpleNamespace(parentIndex="3")
    assert resolve_tmdb_season_number(None, episode) == 3


def test_resolve_tmdb_season_number_parent_title_digit():
    episode = types.SimpleNamespace(parentTitle="4")
    assert resolve_tmdb_season_number(None, episode) == 4


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


def test_fetch_imdb_batch_http_error(monkeypatch):
    monkeypatch.setattr(loader, "_imdb_cache", None)

    async def raise_error(request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        raise httpx.ConnectError("boom", request=request)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(raise_error)) as client:
            result = await _fetch_imdb_batch(client, ["tt1", "tt2"])
            assert result == {"tt1": None, "tt2": None}

    asyncio.run(main())


def test_fetch_imdb_batch_rate_limited(monkeypatch):
    retry_queue = loader._IMDbRetryQueue()
    monkeypatch.setattr(loader, "_imdb_retry_queue", retry_queue)
    monkeypatch.setattr(loader, "_imdb_max_retries", 1)
    monkeypatch.setattr(loader, "_imdb_backoff", 0.01)

    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def rate_limited(request: httpx.Request) -> httpx.Response:  # type: ignore[override]
        return httpx.Response(429)

    async def main() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(rate_limited)) as client:
            result = await _fetch_imdb_batch(client, ["tt3"])
            assert result["tt3"] is None

    asyncio.run(main())

    assert sleeps == [0.01]
    assert retry_queue.snapshot() == ["tt3"]


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


def test_build_plex_item_converts_string_indices():
    raw = types.SimpleNamespace(
        ratingKey="1",
        guid="g",
        type="episode",
        title="Episode",
        parentIndex="02",
        index="03",
    )

    item = _build_plex_item(raw)
    assert item.season_number == 2
    assert item.episode_number == 3


def test_run_live_loader_builds_payload_with_collections(monkeypatch):
    monkeypatch.setattr(loader, "_imdb_cache", None)
    monkeypatch.setattr(loader, "_imdb_retry_queue", loader._IMDbRetryQueue())

    class DummyClient:
        def __init__(self, *args, **kwargs):
            self._client = None

        async def collection_exists(self, collection_name: str) -> bool:
            return False

        async def create_collection(self, *args, **kwargs) -> None:
            pass

        async def create_payload_index(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setattr(loader, "AsyncQdrantClient", lambda *args, **kwargs: DummyClient())

    async def fake_iter(server, tmdb_api_key, *, batch_size=50):
        plex_item = PlexItem(
            rating_key="1",
            guid="guid",
            type="movie",
            title="Sample",
            summary="Summary",
            year=2024,
            added_at=datetime.fromtimestamp(1),
            guids=[PlexGuid(id="plex://1")],
            thumb="thumb.jpg",
            art="art.jpg",
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
        yield AggregatedItem(plex=plex_item, imdb=imdb_title, tmdb=tmdb_movie)

    monkeypatch.setattr(loader, "_iter_from_plex", fake_iter)

    class DummyPlexServer:
        def __init__(self, url: str, token: str) -> None:
            self.url = url
            self.token = token

    monkeypatch.setattr(loader, "PlexServer", DummyPlexServer)

    recorded_batches: list[list[models.PointStruct]] = []

    async def record_upsert(client, collection_name: str, points):
        recorded_batches.append(list(points))

    monkeypatch.setattr(loader, "_upsert_in_batches", record_upsert)

    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)

    asyncio.run(
        loader.run(
            "http://localhost:32400",
            "token",
            "tmdb-key",
            None,
            None,
            None,
        )
    )

    assert recorded_batches, "expected upsert batches to be scheduled"
    payload = recorded_batches[0][0].payload
    assert payload["collections"] == ["Favorites"]
    assert payload["reviews"] == ["Great"]
