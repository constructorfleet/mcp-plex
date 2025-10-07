import asyncio
import runpy
import sys

import pytest
from click.testing import CliRunner

from mcp_plex import loader
from mcp_plex.loader import cli as loader_cli


def test_cli_continuous_respects_delay(monkeypatch):
    actions: list = []
    run_calls = 0

    async def fake_run(*args, **kwargs):
        nonlocal run_calls
        run_calls += 1
        actions.append("run")
        if run_calls >= 2:
            raise RuntimeError("stop")

    async def fake_sleep(seconds):
        actions.append(("sleep", seconds))

    monkeypatch.setattr(loader, "run", fake_run)
    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    runner = CliRunner()
    with pytest.raises(RuntimeError, match="stop"):
        runner.invoke(
            loader_cli.main,
            ["--continuous", "--delay", "7.5"],
            catch_exceptions=False,
            env={
                "PLEX_URL": "http://localhost",
                "PLEX_TOKEN": "token",
                "TMDB_API_KEY": "key",
            },
        )

    assert actions == ["run", ("sleep", 7.5), "run"]


def test_cli_invalid_delay_value():
    runner = CliRunner()
    result = runner.invoke(
        loader_cli.main,
        ["--delay", "not-a-number"],
        env={
            "PLEX_URL": "http://localhost",
            "PLEX_TOKEN": "token",
            "TMDB_API_KEY": "key",
        },
    )
    assert result.exit_code != 0
    assert "Invalid value for '--delay'" in result.output


def test_run_requires_credentials(monkeypatch):
    monkeypatch.setattr(loader, "PlexServer", object)

    async def invoke():
        await loader.run(None, None, "key", None, None, None)

    with pytest.raises(RuntimeError, match="PLEX_URL and PLEX_TOKEN must be provided"):
        asyncio.run(invoke())


def test_run_requires_tmdb_api_key(monkeypatch):
    monkeypatch.setattr(loader, "PlexServer", object)

    async def invoke():
        await loader.run("http://localhost", "token", None, None, None, None)

    with pytest.raises(RuntimeError, match="TMDB_API_KEY must be provided"):
        asyncio.run(invoke())


def test_run_requires_plexapi(monkeypatch):
    monkeypatch.setattr(loader, "PlexServer", None)

    async def invoke():
        await loader.run("http://localhost", "token", "key", None, None, None)

    with pytest.raises(RuntimeError, match="plexapi is required for live loading"):
        asyncio.run(invoke())


def test_cli_model_overrides(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_run(**kwargs):
        captured["dense"] = kwargs["dense_model_name"]
        captured["sparse"] = kwargs["sparse_model_name"]

    monkeypatch.setattr(loader, "run", fake_run)

    runner = CliRunner()
    runner.invoke(
        loader_cli.main,
        ["--dense-model", "foo", "--sparse-model", "bar"],
        catch_exceptions=False,
        env={
            "PLEX_URL": "http://localhost",
            "PLEX_TOKEN": "token",
            "TMDB_API_KEY": "key",
        },
    )

    assert captured["dense"] == "foo"
    assert captured["sparse"] == "bar"


def test_cli_model_env(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_run(**kwargs):
        captured["dense"] = kwargs["dense_model_name"]
        captured["sparse"] = kwargs["sparse_model_name"]

    monkeypatch.setattr(loader, "run", fake_run)

    runner = CliRunner()
    runner.invoke(
        loader_cli.main,
        [],
        catch_exceptions=False,
        env={
            "PLEX_URL": "http://localhost",
            "PLEX_TOKEN": "token",
            "TMDB_API_KEY": "key",
            "DENSE_MODEL": "foo",
            "SPARSE_MODEL": "bar",
        },
    )

    assert captured["dense"] == "foo"
    assert captured["sparse"] == "bar"


def test_load_media_passes_imdb_queue_path(monkeypatch, tmp_path):
    imdb_queue = tmp_path / "queue.json"
    imdb_cache = tmp_path / "cache.json"

    captured_kwargs: dict[str, object] = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(loader, "run", fake_run)

    asyncio.run(
        loader.load_media(
            plex_url="http://localhost",
            plex_token="token",
            tmdb_api_key="key",
            sample_dir=None,
            qdrant_url=":memory:",
            qdrant_api_key=None,
            qdrant_host=None,
            qdrant_port=6333,
            qdrant_grpc_port=6334,
            qdrant_https=False,
            qdrant_prefer_grpc=False,
            dense_model_name="dense",
            sparse_model_name="sparse",
            continuous=False,
            delay=0.0,
            imdb_cache=imdb_cache,
            imdb_max_retries=3,
            imdb_backoff=1.0,
            imdb_requests_per_window=None,
            imdb_window_seconds=1.0,
            imdb_queue=imdb_queue,
            upsert_buffer_size=1,
            plex_chunk_size=1,
            enrichment_batch_size=1,
            enrichment_workers=1,
            qdrant_batch_size=1,
            max_concurrent_upserts=1,
            qdrant_retry_attempts=1,
            qdrant_retry_backoff=1.0,
        )
    )

    assert captured_kwargs["imdb_queue_path"] == imdb_queue
    assert captured_kwargs["imdb_cache_path"] == imdb_cache


def test_loader_script_entrypoint(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["loader", "--help"])
    module = sys.modules.pop("mcp_plex.loader.cli", None)
    try:
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("mcp_plex.loader.cli", run_name="__main__")
    finally:
        if module is not None:
            sys.modules["mcp_plex.loader.cli"] = module
    assert exc.value.code == 0
