from unittest.mock import patch

import asyncio
import importlib
import pytest

from mcp_plex import server as server_package
from mcp_plex.server import cli as server


@pytest.fixture(scope="module", autouse=True)
def close_server_module():
    yield
    asyncio.run(server.server.close())


def test_main_stdio_runs():
    with patch.object(server.server, "run") as mock_run:
        server.main([])
        mock_run.assert_called_once_with(transport="stdio")


def test_main_requires_bind_and_port_for_http():
    with pytest.raises(SystemExit):
        server.main(["--transport", "sse", "--bind", "0.0.0.0"])
    with pytest.raises(SystemExit):
        server.main(["--transport", "sse", "--port", "8000"])


def test_main_mount_disallowed_for_stdio():
    with pytest.raises(SystemExit):
        server.main(["--mount", "/mcp"])


def test_main_http_with_mount_runs():
    with patch.object(server.server, "run") as mock_run:
        server.main(["--transport", "sse", "--bind", "0.0.0.0", "--port", "8000", "--mount", "/mcp"])
        mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=8000, path="/mcp")


def test_main_model_overrides():
    with patch.object(server.server, "run") as mock_run:
        server.main([
            "--dense-model",
            "foo",
            "--sparse-model",
            "bar",
        ])
        assert server.settings.dense_model == "foo"
        assert server.settings.sparse_model == "bar"
        mock_run.assert_called_once_with(transport="stdio")


def test_env_model_overrides(monkeypatch):
    monkeypatch.setenv("DENSE_MODEL", "foo")
    monkeypatch.setenv("SPARSE_MODEL", "bar")
    asyncio.run(server.server.close())
    importlib.reload(server_package)
    importlib.reload(server)
    assert server.settings.dense_model == "foo"
    assert server.settings.sparse_model == "bar"

    # reload to reset globals
    asyncio.run(server.server.close())
    importlib.reload(server_package)
    importlib.reload(server)


def test_env_overrides_cli_arguments(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "sse")
    monkeypatch.setenv("MCP_HOST", "1.2.3.4")
    monkeypatch.setenv("MCP_PORT", "1234")
    monkeypatch.setenv("MCP_MOUNT", "/env")
    with patch.object(server.server, "run") as mock_run:
        server.main(
            [
                "--transport",
                "stdio",
                "--bind",
                "0.0.0.0",
                "--port",
                "9999",
                "--mount",
                "/cli",
            ]
        )
        mock_run.assert_called_once_with(
            transport="sse", host="1.2.3.4", port=1234, path="/env"
        )


def test_env_only_http_configuration(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "sse")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MCP_PORT", "8000")
    with patch.object(server.server, "run") as mock_run:
        server.main([])
        mock_run.assert_called_once_with(
            transport="sse", host="0.0.0.0", port=8000
        )


def test_env_invalid_port(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "sse")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MCP_PORT", "not-a-port")
    with pytest.raises(SystemExit):
        server.main([])


def test_run_config_reexport():
    from mcp_plex.server import RunConfig as ExportedRunConfig

    assert ExportedRunConfig is server.RunConfig
