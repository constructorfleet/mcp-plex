from __future__ import annotations

import asyncio
import importlib
import importlib.metadata as metadata
from types import SimpleNamespace

import pytest
from plexapi.exceptions import PlexApiException

from mcp_plex import server as server_module
from mcp_plex.server import PlexServer


def test_version_fallback(monkeypatch):
    def _raise(*args, **kwargs):
        raise metadata.PackageNotFoundError

    monkeypatch.setattr(metadata, "version", _raise)
    reloaded = importlib.reload(server_module)
    try:
        assert reloaded.__version__ == "0.0.0"
    finally:
        importlib.reload(reloaded)


def test_clear_plex_identity_cache_resets_state():
    class DummyClient:
        async def close(self) -> None:
            return None

    settings = server_module.server.settings.model_copy(
        update={"qdrant_url": ":memory:"}
    )
    plex_server = PlexServer(settings=settings, qdrant_client=DummyClient())
    plex_server._plex_identity = {"machineIdentifier": "abc"}
    plex_server._plex_client = SimpleNamespace()

    plex_server.clear_plex_identity_cache()

    assert plex_server._plex_identity is None
    assert plex_server._plex_client is None


def test_request_model_skips_variadic_params():
    def _callable(arg, *args, **kwargs):
        return arg

    model = server_module._request_model("test", _callable)
    assert model is not None
    assert list(model.model_fields) == ["arg"]


def test_request_model_returns_none_for_no_params():
    def _empty():
        return None

    assert server_module._request_model("noop", _empty) is None


def test_ensure_plex_configuration_requires_settings(monkeypatch):
    original_settings = server_module.server.settings
    modified = original_settings.model_copy(
        update={"plex_url": None, "plex_token": None}
    )
    monkeypatch.setattr(server_module.server, "_settings", modified)
    try:
        with pytest.raises(RuntimeError):
            server_module._ensure_plex_configuration()
    finally:
        monkeypatch.setattr(server_module.server, "_settings", original_settings)


def test_fetch_plex_identity_requires_identifier(monkeypatch):
    async def _fake_get_client():
        return SimpleNamespace(machineIdentifier=None)

    monkeypatch.setattr(server_module, "_get_plex_client", _fake_get_client)
    server_module.server._plex_identity = None

    with pytest.raises(RuntimeError):
        asyncio.run(server_module._fetch_plex_identity())


def test_get_plex_players_handles_non_iterable_provides(monkeypatch):
    class StubClient:
        provides = 123
        machineIdentifier = "machine"
        clientIdentifier = "client"
        address = "127.0.0.1"
        port = "32400"
        title = "Living Room"
        product = "Plex"

    async def _fake_get_client():
        return SimpleNamespace(clients=lambda: [StubClient()])

    monkeypatch.setattr(server_module, "_get_plex_client", _fake_get_client)
    server_module.server._plex_client = None

    players = asyncio.run(server_module._get_plex_players())
    assert players[0]["display_name"] == "Living Room"


def test_match_player_skips_blank_candidates():
    player = {
        "display_name": "Living Room",
        "friendly_names": ["", "Lounge"],
        "provides": {"player"},
    }

    selected = server_module._match_player("lounge", [player])
    assert selected["display_name"] == "Living Room"


def test_start_playback_requires_client():
    with pytest.raises(ValueError):
        asyncio.run(
            server_module._start_playback(
                "1",
                {"display_name": "Living Room", "provides": {"player"}},
                0,
            )
        )


def test_start_playback_wraps_plex_errors(monkeypatch):
    class PlexClientStub:
        def playMedia(self, *args, **kwargs):
            raise PlexApiException("boom")

    async def _fake_get_client():
        return SimpleNamespace(fetchItem=lambda path: object())

    async def _fake_fetch_identity():
        return {"machineIdentifier": "abc"}

    monkeypatch.setattr(server_module, "_get_plex_client", _fake_get_client)
    monkeypatch.setattr(server_module, "_fetch_plex_identity", _fake_fetch_identity)

    with pytest.raises(RuntimeError):
        asyncio.run(
            server_module._start_playback(
                "1",
                {
                    "display_name": "Living Room",
                    "provides": {"player"},
                    "client": PlexClientStub(),
                },
                0,
            )
        )


def test_server_main_invokes_cli(monkeypatch):
    captured: dict[str, object] = {}

    def fake_main(argv=None):
        captured["argv"] = argv

    from mcp_plex.server import cli as server_cli

    monkeypatch.setattr(server_cli, "main", fake_main)
    server_module.main(["--help"])
    assert captured["argv"] == ["--help"]


def test_module_getattr_exposes_runconfig():
    from mcp_plex.server.cli import RunConfig

    assert server_module.RunConfig is RunConfig
