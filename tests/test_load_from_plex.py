import asyncio
import types

import httpx

from mcp_plex import loader
from mcp_plex.types import TMDBShow


def test_load_from_plex(monkeypatch):
    movie = types.SimpleNamespace(
        ratingKey="101",
        guid="plex://movie/101",
        type="movie",
        title="Inception",
        guids=[
            types.SimpleNamespace(id="imdb://tt1375666"),
            types.SimpleNamespace(id="tmdb://27205"),
        ],
    )

    ep1 = types.SimpleNamespace(
        ratingKey="102",
        guid="plex://episode/102",
        type="episode",
        title="Pilot",
        guids=[
            types.SimpleNamespace(id="imdb://tt0959621"),
            types.SimpleNamespace(id="tmdb://62085"),
        ],
    )
    ep2 = types.SimpleNamespace(
        ratingKey="103",
        guid="plex://episode/103",
        type="episode",
        title="Cat's in the Bag...",
        guids=[types.SimpleNamespace(id="imdb://tt0959622")],
    )

    show = types.SimpleNamespace(
        guids=[types.SimpleNamespace(id="tmdb://1396")],
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
        if "tt1375666" in url:
            return httpx.Response(
                200,
                json={
                    "id": "tt1375666",
                    "type": "movie",
                    "primaryTitle": "Inception",
                },
            )
        if "tt0959621" in url:
            return httpx.Response(
                200,
                json={
                    "id": "tt0959621",
                    "type": "episode",
                    "primaryTitle": "Pilot",
                },
            )
        if "tt0959622" in url:
            return httpx.Response(
                200,
                json={
                    "id": "tt0959622",
                    "type": "episode",
                    "primaryTitle": "Cat's in the Bag...",
                },
            )
        if "/movie/27205" in url:
            return httpx.Response(200, json={"id": 27205, "title": "Inception"})
        if "/tv/1396" in url:
            return httpx.Response(200, json={"id": 1396, "name": "Breaking Bad"})
        if "/episode/62085" in url:
            return httpx.Response(200, json={"id": 62085, "name": "Pilot"})
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
    assert items[0].tmdb and items[0].tmdb.id == 27205
    assert items[1].tmdb and items[1].tmdb.id == 62085
    assert isinstance(items[2].tmdb, TMDBShow)
    assert items[2].tmdb.id == 1396
