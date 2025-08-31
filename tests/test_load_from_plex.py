import asyncio
import types
import httpx

from mcp_plex import loader
from mcp_plex.types import TMDBShow


def test_load_from_plex(monkeypatch):
    movie = types.SimpleNamespace(
        ratingKey="1",
        guid="g1",
        type="movie",
        title="Movie",
        guids=[
            types.SimpleNamespace(id="imdb://ttm"),
            types.SimpleNamespace(id="tmdb://1"),
        ],
    )

    ep1 = types.SimpleNamespace(
        ratingKey="2",
        guid="g2",
        type="episode",
        title="Ep1",
        guids=[
            types.SimpleNamespace(id="imdb://tt1"),
            types.SimpleNamespace(id="tmdb://2"),
        ],
    )
    ep2 = types.SimpleNamespace(
        ratingKey="3",
        guid="g3",
        type="episode",
        title="Ep2",
        guids=[types.SimpleNamespace(id="imdb://tt2")],
    )

    show = types.SimpleNamespace(
        guids=[types.SimpleNamespace(id="tmdb://3")],
        episodes=lambda: [ep1, ep2],
    )

    movie_section = types.SimpleNamespace(all=lambda: [movie])
    show_section = types.SimpleNamespace(all=lambda: [show])
    library = types.SimpleNamespace(
        section=lambda name: movie_section if name == "Movies" else show_section
    )
    server = types.SimpleNamespace(library=library)

    async def handler(request):
        url = str(request.url)
        if "imdbapi" in url:
            return httpx.Response(
                200, json={"id": "tt", "type": "movie", "primaryTitle": "IMDb"}
            )
        if "/movie/1" in url:
            return httpx.Response(200, json={"id": 1, "title": "TMDB Movie"})
        if "/tv/3" in url:
            return httpx.Response(200, json={"id": 3, "name": "TMDB Show"})
        if "/episode/2" in url:
            return httpx.Response(200, json={"id": 2, "name": "TMDB Ep"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient
    monkeypatch.setattr(
        loader.httpx,
        "AsyncClient",
        lambda *args, **kwargs: orig_client(transport=transport),
    )

    items = asyncio.run(loader._load_from_plex(server, "key"))
    assert len(items) == 3
    assert items[0].tmdb and items[0].tmdb.id == 1
    assert items[1].tmdb and items[1].tmdb.id == 2
    assert isinstance(items[2].tmdb, TMDBShow)
    assert items[2].tmdb.id == 3
