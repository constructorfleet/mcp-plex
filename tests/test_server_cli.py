from unittest.mock import patch

import pytest

from mcp_plex import server


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
        assert server._DENSE_MODEL_NAME == "foo"
        assert server._SPARSE_MODEL_NAME == "bar"
        mock_run.assert_called_once_with(transport="stdio")


def test_env_model_overrides(monkeypatch):
    monkeypatch.setenv("DENSE_MODEL", "foo")
    monkeypatch.setenv("SPARSE_MODEL", "bar")
    import importlib

    importlib.reload(server)
    assert server._DENSE_MODEL_NAME == "foo"
    assert server._SPARSE_MODEL_NAME == "bar"

    # reload to reset globals
    importlib.reload(server)
