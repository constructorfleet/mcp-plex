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
        parentIndex=1,
        index=1,
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
        ratingKey="201",
        guids=[types.SimpleNamespace(id="tmdb://1396")],
        episodes=lambda: [ep1, ep2],
    )

    movie_section = types.SimpleNamespace(all=lambda: [movie])
    show_section = types.SimpleNamespace(all=lambda: [show])
    library = types.SimpleNamespace(
        section=lambda name: movie_section if name == "Movies" else show_section
    )

    items = {101: movie, 102: ep1, 103: ep2, 201: show}

    def fetchItems(keys):
        return [items[int(k)] for k in keys]

    server = types.SimpleNamespace(library=library, fetchItems=fetchItems)

    async def handler(request):
        url = str(request.url)
        if "themoviedb.org" in url:
            assert request.headers.get("Authorization") == "Bearer key"
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
        if "/tv/1396/season/1/episode/1" in url:
            return httpx.Response(200, json={"id": 62085, "name": "Pilot"})
        if "/tv/1396" in url:
            return httpx.Response(200, json={"id": 1396, "name": "Breaking Bad"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    orig_client = httpx.AsyncClient
    monkeypatch.setattr(
        loader.httpx,
        "AsyncClient",
        lambda *args, **kwargs: orig_client(transport=transport),
    )

    calls = []
    orig_batch = loader._gather_in_batches

    async def fake_batch(tasks, batch_size):
        calls.append((len(tasks), batch_size))
        return await orig_batch(tasks, batch_size)

    monkeypatch.setattr(loader, "_gather_in_batches", fake_batch)

    items = asyncio.run(loader._load_from_plex(server, "key", batch_size=1))
    assert calls == [(1, 1), (2, 1)]
    assert len(items) == 3
    assert items[0].tmdb and items[0].tmdb.id == 27205
    assert items[1].tmdb and items[1].tmdb.id == 62085
    assert isinstance(items[2].tmdb, TMDBShow)
    assert items[2].tmdb.id == 1396
