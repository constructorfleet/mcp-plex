from __future__ import annotations

import asyncio

import pytest

from mcp_plex import loader


def test_imdb_runtime_config_creates_throttle():
    config = loader.IMDbRuntimeConfig(
        cache=None,
        max_retries=1,
        backoff=1.0,
        retry_queue=loader.IMDbRetryQueue(),
        requests_per_window=2,
        window_seconds=3.0,
    )

    throttle = config.get_throttle()
    assert throttle is not None
    assert config.get_throttle() is throttle


class _DummyClient:
    async def close(self) -> None:
        return None


def test_run_requires_plex_configuration(monkeypatch):
    async def _run() -> None:
        monkeypatch.setattr(loader, "AsyncQdrantClient", lambda *a, **k: _DummyClient())

        async def _noop(*args, **kwargs):
            return None

        monkeypatch.setattr(loader, "_ensure_collection", _noop)

        with pytest.raises(
            RuntimeError, match="PLEX_URL and PLEX_TOKEN must be provided"
        ):
            await loader.run(
                plex_url=None,
                plex_token=None,
                tmdb_api_key=None,
                sample_dir=None,
                qdrant_url=None,
                qdrant_api_key=None,
            )

    asyncio.run(_run())
