from __future__ import annotations

from pathlib import Path
import runpy


ENTRYPOINT = Path("entrypoint.sh")
VENV_BIN = "/app/.venv/bin"


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


def test_docker_entrypoint_executes_virtualenv_commands() -> None:
    """Ensure the container entrypoint dispatches through the prepared virtualenv."""

    contents = ENTRYPOINT.read_text(encoding="utf-8")

    assert (
        f'"{VENV_BIN}/$command"' in contents
    ), "Entrypoint should execute commands from the virtualenv bin directory"
    assert "uv run" not in contents, "Entrypoint should not invoke uv run during dispatch"
