import types
import asyncio
import httpx

from pathlib import Path

from mcp_plex.loader import (
    _extract_external_ids,
    _load_from_sample,
    _build_plex_item,
    _fetch_imdb,
    _fetch_tmdb_movie,
    _fetch_tmdb_show,
    _fetch_tmdb_episode,
)


def test_extract_external_ids():
    guid_objs = [
        types.SimpleNamespace(id="imdb://tt0133093"),
        types.SimpleNamespace(id="tmdb://603"),
    ]
    item = types.SimpleNamespace(guids=guid_objs)
    ids = _extract_external_ids(item)
    assert ids.imdb == "tt0133093"
    assert ids.tmdb == "603"


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


def test_fetch_functions_success_and_failure():
    async def imdb_mock(request):
        if "good" in str(request.url):
            return httpx.Response(200, json={"id": "tt1", "type": "movie", "primaryTitle": "T"})
        return httpx.Response(404)

    async def tmdb_movie_mock(request):
        if "good" in str(request.url):
            return httpx.Response(200, json={"id": 1, "title": "M"})
        return httpx.Response(404)

    async def tmdb_show_mock(request):
        if "good" in str(request.url):
            return httpx.Response(200, json={"id": 1, "name": "S"})
        return httpx.Response(404)

    async def tmdb_episode_mock(request):
        if "good" in str(request.url):
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
            assert (await _fetch_tmdb_episode(client, "good", "k")) is not None
            assert (await _fetch_tmdb_episode(client, "bad", "k")) is None

    asyncio.run(main())
