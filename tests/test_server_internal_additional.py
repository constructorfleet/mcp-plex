from __future__ import annotations

import asyncio
import importlib
import importlib.metadata as metadata
import json
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from plexapi.exceptions import PlexApiException
from plexapi import base as plex_base

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


def test_qdrant_client_reinitializes_after_close(monkeypatch):
    from qdrant_client import async_qdrant_client

    instances: list[object] = []

    class StubClient:
        def __init__(self, *args, **kwargs):
            instances.append(self)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(async_qdrant_client, "AsyncQdrantClient", StubClient)
    reloaded = importlib.reload(server_module)
    try:
        first = reloaded.server.qdrant_client
        asyncio.run(reloaded.server.close())
        second = reloaded.server.qdrant_client
        assert first is not second
        assert len(instances) >= 2
    finally:
        asyncio.run(reloaded.server.close())
        importlib.reload(reloaded)


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


def test_get_plex_players_supports_alias_key_lookup(monkeypatch):
    class StubClient:
        provides = "player"
        machineIdentifier = "machine-1"
        clientIdentifier = "client-1"
        address = "192.0.2.10"
        port = 32400
        title = "Movie Room Player"
        product = "Plex"

    async def _fake_get_client():
        return SimpleNamespace(clients=lambda: [StubClient()])

    original_settings = server_module.server.settings
    updated_settings = original_settings.model_copy(
        update={"plex_player_aliases": {"Movie Room": ("machine-1",)}}
    )

    monkeypatch.setattr(server_module, "_get_plex_client", _fake_get_client)
    monkeypatch.setattr(server_module.server, "_settings", updated_settings)
    server_module.server._plex_client = None

    try:
        players = asyncio.run(server_module._get_plex_players())
    finally:
        monkeypatch.setattr(server_module.server, "_settings", original_settings)

    assert players[0]["friendly_names"] == ["Movie Room"]
    assert players[0]["display_name"] == "Movie Room"
    matched = server_module._match_player("movie room", players)
    assert matched is players[0]


def test_get_plex_players_uses_fixture_when_available(monkeypatch, tmp_path):
    xml_payload = """
    <MediaContainer size="2">
        <Server name="Apple TV" host="10.0.12.122" address="10.0.12.122" port="32500" machineIdentifier="243795C0-C395-4C64-AFD9-E12390C86595" version="8.45" protocol="plex" product="Plex for Apple TV" deviceClass="stb" protocolVersion="2" protocolCapabilities="playback,playqueues,timeline,provider-playback">
            <Alias>Movie Room TV</Alias>
            <Alias>Movie Room</Alias>
        </Server>
        <Server name="Apple TV" host="10.0.12.94" address="10.0.12.94" port="32500" machineIdentifier="243795C0-C395-4C64-AFD9-E12390C86212" version="8.45" protocol="plex" product="Plex for Apple TV" deviceClass="stb" protocolVersion="2" protocolCapabilities="playback,playqueues,timeline,provider-playback">
            <Alias>Office AppleTV</Alias>
            <Alias>Office TV</Alias>
            <Alias>Office</Alias>
        </Server>
    </MediaContainer>
    """
    payload_path = tmp_path / "clients.xml"
    payload_path.write_text(xml_payload, encoding="utf-8")

    original_settings = server_module.server.settings
    updated_settings = original_settings.model_copy(
        update={"plex_clients_file": payload_path}
    )
    monkeypatch.setattr(server_module.server, "_settings", updated_settings)

    calls: list[str] = []

    async def _unexpected_call() -> None:
        calls.append("clients")
        return SimpleNamespace(clients=lambda: [])

    monkeypatch.setattr(server_module, "_get_plex_client", _unexpected_call)

    created: list[SimpleNamespace] = []
    load_calls: list[ElementTree.Element] = []

    class StubPlexClient:
        def __init__(self, **kwargs):
            data = kwargs.pop("data", None)
            created.append(SimpleNamespace(**kwargs))
            for key, value in kwargs.items():
                setattr(self, key, value)
            if data is not None:
                self._loadData(data)

        def _loadData(self, element: ElementTree.Element) -> None:
            load_calls.append(element)
            self.machineIdentifier = element.attrib.get("machineIdentifier")
            self.clientIdentifier = element.attrib.get("clientIdentifier")
            title_value = element.attrib.get("title") or element.attrib.get("name")
            if title_value:
                self.title = title_value
                self.name = title_value

    monkeypatch.setattr(server_module, "PlexClient", StubPlexClient)

    try:
        players = asyncio.run(server_module._get_plex_players())
    finally:
        monkeypatch.setattr(server_module.server, "_settings", original_settings)

    assert not calls
    assert len(players) == 2
    assert players[0]["display_name"] == "Movie Room TV"
    assert players[0]["friendly_names"] == ["Movie Room TV", "Movie Room"]
    assert players[0]["provides"] == {
        "playback",
        "playqueues",
        "timeline",
        "provider-playback",
    }
    assert created[0].baseurl == "http://10.0.12.122:32500"
    assert created[0].identifier == "243795C0-C395-4C64-AFD9-E12390C86595"
    assert load_calls
    first_loaded = load_calls[0]
    assert first_loaded.tag == "Server"
    assert first_loaded.attrib.get("protocolCapabilities") == (
        "playback,playqueues,timeline,provider-playback"
    )
    assert [
        alias.text for alias in first_loaded.findall("Alias")
    ] == ["Movie Room TV", "Movie Room"]


def test_collect_session_players_uses_configured_client_for_identifier(
    monkeypatch, tmp_path
):
    payload = [
        {
            "machineIdentifier": "machine-42",
            "clientIdentifier": "client-99",
            "name": "Movie Nook",
            "host": "10.0.0.9",
            "port": 32400,
        }
    ]
    payload_path = tmp_path / "clients.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    original_settings = server_module.server.settings
    updated_settings = original_settings.model_copy(
        update={"plex_clients_file": payload_path}
    )
    monkeypatch.setattr(server_module.server, "_settings", updated_settings)

    created: list[SimpleNamespace] = []

    class StubPlexClient(SimpleNamespace):
        def __init__(self, **kwargs):
            data = kwargs.pop("data", None)
            created.append(SimpleNamespace(**kwargs))
            super().__init__(**kwargs)
            if data is not None:
                self._loadData(data)

        def _loadData(self, element: ElementTree.Element) -> None:
            self.machineIdentifier = element.attrib.get("machineIdentifier")
            self.clientIdentifier = element.attrib.get("clientIdentifier")
            title_value = element.attrib.get("title") or element.attrib.get("name")
            if title_value:
                self.title = title_value
                self.name = title_value

    monkeypatch.setattr(server_module, "PlexClient", StubPlexClient)

    try:
        session = SimpleNamespace(player="client-99")
        players = server_module._collect_session_players(session)
    finally:
        monkeypatch.setattr(server_module.server, "_settings", original_settings)

    assert len(players) == 1
    selected = players[0]
    assert getattr(selected, "identifier", None) == "client-99"
    assert getattr(selected, "machineIdentifier", None) == "machine-42"
    assert getattr(selected, "title", None) == "Movie Nook"


def test_plex_session_player_returns_configured_client(monkeypatch, tmp_path):
    payload = [
        {
            "machineIdentifier": "machine-42",
            "clientIdentifier": "client-99",
            "name": "Movie Nook",
            "host": "10.0.0.9",
            "port": 32400,
        }
    ]
    payload_path = tmp_path / "clients.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    original_settings = server_module.server.settings
    updated_settings = original_settings.model_copy(
        update={"plex_clients_file": payload_path}
    )
    monkeypatch.setattr(server_module.server, "_settings", updated_settings)

    created: list[SimpleNamespace] = []

    class StubPlexClient(SimpleNamespace):
        def __init__(self, **kwargs):
            data = kwargs.pop("data", None)
            created.append(SimpleNamespace(**kwargs))
            super().__init__(**kwargs)
            if data is not None:
                self._loadData(data)

        def _loadData(self, element: ElementTree.Element) -> None:
            self.machineIdentifier = element.attrib.get("machineIdentifier")
            self.clientIdentifier = element.attrib.get("clientIdentifier")
            title_value = element.attrib.get("title") or element.attrib.get("name")
            if title_value:
                self.title = title_value
                self.name = title_value

    monkeypatch.setattr(server_module, "PlexClient", StubPlexClient)

    class DummySession(plex_base.PlexSession):
        def __init__(self):
            self._data = object()

        def findItem(self, data, etag=None, **_kwargs):
            return "client-99"

    try:
        session = DummySession()
        player = session.player
    finally:
        monkeypatch.setattr(server_module.server, "_settings", original_settings)

    assert isinstance(player, StubPlexClient)
    assert getattr(player, "identifier", None) == "client-99"
    assert getattr(player, "machineIdentifier", None) == "machine-42"
    assert created


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


def test_start_playback_allows_players_without_player_capability(monkeypatch):
    class PlexClientStub:
        def __init__(self):
            self.calls: list[tuple[object, int, str]] = []

        def playMedia(self, media, *, offset, machineIdentifier):
            self.calls.append((media, offset, machineIdentifier))

    class PlexServerStub:
        def fetchItem(self, path):
            return {"path": path}

    plex_client = PlexClientStub()

    async def _fake_get_client():
        return PlexServerStub()

    async def _fake_fetch_identity():
        return {"machineIdentifier": "abc"}

    monkeypatch.setattr(server_module, "_get_plex_client", _fake_get_client)
    monkeypatch.setattr(server_module, "_fetch_plex_identity", _fake_fetch_identity)

    asyncio.run(
        server_module._start_playback(
            "42",
            {"display_name": "Living Room", "client": plex_client, "provides": set()},
            5,
        )
    )

    assert plex_client.calls == [({"path": "/library/metadata/42"}, 5000, "abc")]


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
