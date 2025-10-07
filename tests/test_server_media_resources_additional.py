from __future__ import annotations

import asyncio

import pytest

from mcp_plex.server import media as media_helpers
from mcp_plex.server import media_background, media_poster


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
