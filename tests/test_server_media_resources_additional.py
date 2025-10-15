from __future__ import annotations

import asyncio

import pytest

from mcp_plex.server import media as media_helpers
from mcp_plex.server import media_background, media_poster, server as plex_server


def test_media_poster_raises_when_missing(monkeypatch):
    async def _run() -> None:
        async def _fake_get_media(*args, **kwargs):
            return {"plex": {"rating_key": "missing"}}

        monkeypatch.setattr(media_helpers, "_get_media_data", _fake_get_media)

        with pytest.raises(ValueError, match="Poster not available"):
            await media_poster.fn(identifier="missing")

    asyncio.run(_run())


def test_media_background_raises_when_missing(monkeypatch):
    async def _run() -> None:
        async def _fake_get_media(*args, **kwargs):
            return {"plex": {"rating_key": "missing"}}

        monkeypatch.setattr(media_helpers, "_get_media_data", _fake_get_media)

        with pytest.raises(ValueError, match="Background not available"):
            await media_background.fn(identifier="missing")

    asyncio.run(_run())


def test_media_artwork_cached_for_alternate_identifier(monkeypatch):
    async def _run() -> None:
        plex_server.cache.clear()

        calls: list[str] = []

        rating_key = "49915"
        imdb_id = "tt8367814"
        poster_url = "https://example.com/poster.jpg"
        background_url = "https://example.com/background.jpg"

        async def _fake_get_media(server, identifier):
            calls.append(identifier)
            return {
                "title": "The Gentlemen",
                "imdb": {"id": imdb_id},
                "plex": {
                    "rating_key": rating_key,
                    "thumb": poster_url,
                    "art": background_url,
                },
            }

        monkeypatch.setattr(media_helpers, "_get_media_data", _fake_get_media)

        poster = await media_poster.fn(identifier=rating_key)
        assert poster == poster_url
        poster_cached = await media_poster.fn(identifier=imdb_id)
        assert poster_cached == poster_url
        background_cached = await media_background.fn(identifier=imdb_id)
        assert background_cached == background_url
        assert calls == [rating_key]

    asyncio.run(_run())
