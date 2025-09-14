import asyncio
import json
import types
from pathlib import Path

import httpx

from mcp_plex import loader
from mcp_plex.imdb_cache import IMDbCache
from mcp_plex.loader import (
    _build_plex_item,
    _extract_external_ids,
    _fetch_imdb,
    _fetch_tmdb_episode,
    _fetch_tmdb_movie,
    _fetch_tmdb_show,
    _load_from_sample,
    _load_imdb_retry_queue,
    _persist_imdb_retry_queue,
    _process_imdb_retry_queue,
    resolve_tmdb_season_number,
)
from mcp_plex.types import TMDBSeason, TMDBShow


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
        return httpx.Response(200, json={"id": "tt1", "type": "movie", "primaryTitle": "T"})

    async def first_run():
        _load_imdb_retry_queue(queue_path)
        async with httpx.AsyncClient(transport=httpx.MockTransport(first_transport)) as client:
            await _process_imdb_retry_queue(client)
            await _fetch_imdb(client, "tt1")
        _persist_imdb_retry_queue(queue_path)

    asyncio.run(first_run())
    assert json.loads(queue_path.read_text()) == ["tt1"]

    async def second_run():
        _load_imdb_retry_queue(queue_path)
        async with httpx.AsyncClient(transport=httpx.MockTransport(second_transport)) as client:
            await _process_imdb_retry_queue(client)
        _persist_imdb_retry_queue(queue_path)

    asyncio.run(second_run())
    assert json.loads(queue_path.read_text()) == []
    assert loader._imdb_cache.get("tt1") is not None


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
