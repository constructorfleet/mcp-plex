import asyncio

import pytest
from click.testing import CliRunner

from mcp_plex import loader


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
            loader.main,
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
        loader.main,
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


def test_cli_model_overrides(monkeypatch):
    captured: dict[str, str] = {}

    async def fake_run(*args, **kwargs):
        captured["dense"] = args[11]
        captured["sparse"] = args[12]

    monkeypatch.setattr(loader, "run", fake_run)

    runner = CliRunner()
    runner.invoke(
        loader.main,
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

    async def fake_run(*args, **kwargs):
        captured["dense"] = args[11]
        captured["sparse"] = args[12]

    monkeypatch.setattr(loader, "run", fake_run)

    runner = CliRunner()
    runner.invoke(
        loader.main,
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
