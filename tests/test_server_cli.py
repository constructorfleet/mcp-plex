from unittest.mock import patch

from unittest.mock import patch

import pytest

class _StubDense:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def list_supported_models() -> list[str]:
        return ["stub-dense"]


class _StubSparse:
    def __init__(self, *args, **kwargs) -> None:
        pass

    @staticmethod
    def list_supported_models() -> list[str]:
        return ["stub"]


with patch("fastembed.TextEmbedding", _StubDense), patch(
    "fastembed.SparseTextEmbedding", _StubSparse
):
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
    with patch("mcp_plex.server.TextEmbedding") as mock_dense, patch(
        "mcp_plex.server.SparseTextEmbedding"
    ) as mock_sparse, patch.object(server.server, "run") as mock_run:
        server.main([
            "--dense-model",
            "foo",
            "--sparse-model",
            "bar",
        ])
        mock_dense.assert_called_with("foo")
        mock_sparse.assert_called_with("bar")
        mock_run.assert_called_once_with(transport="stdio")


def test_env_model_overrides(monkeypatch):
    with patch("fastembed.TextEmbedding") as mock_dense, patch(
        "fastembed.SparseTextEmbedding"
    ) as mock_sparse:
        monkeypatch.setenv("DENSE_MODEL", "foo")
        monkeypatch.setenv("SPARSE_MODEL", "bar")
        import importlib

        importlib.reload(server)
        mock_dense.assert_called_with("foo")
        mock_sparse.assert_called_with("bar")
    with patch("fastembed.TextEmbedding"), patch("fastembed.SparseTextEmbedding"):
        import importlib

        importlib.reload(server)
