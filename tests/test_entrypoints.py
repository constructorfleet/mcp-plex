from __future__ import annotations

import runpy


def test_loader_module_entrypoint(monkeypatch):
    captured: dict[str, object] = {}

    def fake_main(argv=None):
        captured["argv"] = argv

    monkeypatch.setattr("mcp_plex.loader.cli.main", fake_main)

    runpy.run_module("mcp_plex.loader", run_name="__main__")

    assert captured["argv"] is None


def test_server_module_entrypoint(monkeypatch):
    captured: dict[str, object] = {}

    def fake_main(argv=None):
        captured["argv"] = argv

    monkeypatch.setattr("mcp_plex.server.cli.main", fake_main)

    runpy.run_module("mcp_plex.server", run_name="__main__")

    assert captured["argv"] is None
