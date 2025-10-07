import asyncio
import types

import httpx

from mcp_plex.loader.pipeline.enrichment import (
    _build_plex_item,
    _extract_external_ids,
    _fetch_tmdb_episode,
    _fetch_tmdb_movie,
    _fetch_tmdb_show,
    resolve_tmdb_season_number,
)
from mcp_plex.common.types import (
    TMDBSeason,
    TMDBShow,
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


def test_extract_external_ids_missing_values():
    item = types.SimpleNamespace(guids=None)
    ids = _extract_external_ids(item)
    assert ids.imdb is None
    assert ids.tmdb is None


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


def test_build_plex_item_normalises_person_ids():
    raw = types.SimpleNamespace(
        ratingKey="1",
        guid="g",
        type="movie",
        title="T",
        directors=[types.SimpleNamespace(id=None, tag="Director")],
        writers=[types.SimpleNamespace(id="5", tag="Writer")],
        actors=[
            types.SimpleNamespace(id="", tag="Actor"),
            types.SimpleNamespace(id="7", tag="Lead"),
        ],
    )

    item = _build_plex_item(raw)
    assert [p.id for p in item.directors] == [0]
    assert [p.id for p in item.writers] == [5]
    assert [p.id for p in item.actors] == [0, 7]


def test_fetch_functions_success_and_failure():
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
        movie_transport = httpx.MockTransport(tmdb_movie_mock)
        show_transport = httpx.MockTransport(tmdb_show_mock)
        episode_transport = httpx.MockTransport(tmdb_episode_mock)

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
            assert await _fetch_tmdb_movie(client, "1", "k") is None
        async with httpx.AsyncClient(transport=transport) as client:
            assert await _fetch_tmdb_show(client, "1", "k") is None
        async with httpx.AsyncClient(transport=transport) as client:
            assert await _fetch_tmdb_episode(client, 1, 1, 1, "k") is None

    asyncio.run(main())


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
