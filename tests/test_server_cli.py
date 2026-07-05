from contextlib import asynccontextmanager
from unittest.mock import patch

import logging
import asyncio
import importlib
import pytest

from mcp_plex import server as server_package
from mcp_plex.server import cli as server
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


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
        server.main(
            [
                "--transport",
                "sse",
                "--bind",
                "0.0.0.0",
                "--port",
                "8000",
                "--mount",
                "/mcp",
            ]
        )
        mock_run.assert_called_once_with(
            transport="sse", host="0.0.0.0", port=8000, path="/mcp"
        )


def test_main_rejects_transport_specific_mounts_for_stdio():
    with pytest.raises(SystemExit):
        server.main(["--sse-mount", "/sse"])
    with pytest.raises(SystemExit):
        server.main(["--streamable-http-mount", "/mcp"])


def test_main_can_run_multiple_http_transports_on_same_port():
    with patch("mcp_plex.server.cli.uvicorn.run") as mock_run:
        server.main(
            [
                "--transport",
                "sse",
                "--transport",
                "streamable-http",
                "--bind",
                "0.0.0.0",
                "--port",
                "8000",
            ]
        )

    mock_run.assert_called_once()
    app = mock_run.call_args.args[0]
    assert mock_run.call_args.kwargs["host"] == "0.0.0.0"
    assert mock_run.call_args.kwargs["port"] == 8000
    assert [route.path for route in app.routes] == ["/sse", "/mcp"]
    assert [getattr(route.app.state, "path", None) for route in app.routes] == ["/", "/"]


def test_shared_http_app_initializes_child_lifespans():
    events: list[str] = []

    def fake_http_app(*, path: str, transport: str) -> Starlette:
        @asynccontextmanager
        async def lifespan(app: Starlette):
            events.append(f"{transport}:start")
            yield
            events.append(f"{transport}:stop")

        app = Starlette(lifespan=lifespan)
        app.state.path = path
        app.state.transport = transport
        return app

    configs = [
        server.HttpTransportConfig(transport="sse", path="/sse"),
        server.HttpTransportConfig(transport="streamable-http", path="/mcp"),
    ]

    with patch.object(server.plex_server, "http_app", side_effect=fake_http_app):
        app = server._build_shared_http_app(configs)

    with TestClient(app):
        pass

    assert events == [
        "sse:start",
        "streamable-http:start",
        "streamable-http:stop",
        "sse:stop",
    ]


def test_shared_http_app_no_redirect_on_exact_mount_path():
    """GET /sse (no trailing slash) must be handled directly, not with a 307 redirect.

    Starlette's ``Mount`` compiles a regex that only matches paths starting with
    ``/sse/`` (trailing slash required), so without the
    ``_TrailingSlashMiddleware`` a plain ``GET /sse`` would trigger Starlette's
    built-in redirect_slashes 307 redirect.  Clients like Home Assistant do not
    follow 307 redirects and would therefore fail to connect.
    """
    from starlette.responses import PlainTextResponse

    def fake_http_app(*, path: str, transport: str) -> Starlette:
        async def root_endpoint(request):
            return PlainTextResponse("ok")

        @asynccontextmanager
        async def lifespan(app: Starlette):
            yield

        child = Starlette(
            lifespan=lifespan,
            routes=[Route("/", endpoint=root_endpoint, methods=["GET"])],
        )
        child.state.path = path
        return child

    configs = [
        server.HttpTransportConfig(transport="sse", path="/sse"),
        server.HttpTransportConfig(transport="streamable-http", path="/mcp"),
    ]

    with patch.object(server.plex_server, "http_app", side_effect=fake_http_app):
        app = server._build_shared_http_app(configs)

    with TestClient(app, follow_redirects=False) as client:
        for mount_path in ("/sse", "/mcp"):
            response = client.get(mount_path)
            assert response.status_code == 200, (
                f"GET {mount_path} returned {response.status_code}; expected 200, not a 307 redirect"
            )


def test_shared_http_app_does_not_rewrite_root_mount():
    from starlette.responses import PlainTextResponse

    observed_paths: list[str] = []

    def fake_http_app(*, path: str, transport: str) -> Starlette:
        async def root_endpoint(request):
            observed_paths.append(request.scope["path"])
            return PlainTextResponse(f"{transport}:{request.scope['path']}")

        @asynccontextmanager
        async def lifespan(app: Starlette):
            yield

        child = Starlette(
            lifespan=lifespan,
            routes=[Route("/", endpoint=root_endpoint, methods=["GET"])],
        )
        child.state.path = path
        return child

    configs = [
        server.HttpTransportConfig(transport="sse", path="/"),
        server.HttpTransportConfig(transport="streamable-http", path="/mcp"),
    ]

    with patch.object(server.plex_server, "http_app", side_effect=fake_http_app):
        app = server._build_shared_http_app(configs)

    with TestClient(app, follow_redirects=False) as client:
        root_response = client.get("/")
        assert root_response.status_code == 200
        assert root_response.text == "sse:/"

        mcp_response = client.get("/mcp")
        assert mcp_response.status_code == 200
        assert mcp_response.text == "streamable-http:/"

    assert observed_paths == ["/", "/"]


def test_shared_http_app_exposes_rest_docs_without_root_mount():
    from starlette.responses import PlainTextResponse

    def fake_http_app(*, path: str, transport: str) -> Starlette:
        async def root_endpoint(request):
            return PlainTextResponse(f"{transport}:{request.scope['path']}")

        @asynccontextmanager
        async def lifespan(app: Starlette):
            yield

        child = Starlette(
            lifespan=lifespan,
            routes=[Route("/", endpoint=root_endpoint, methods=["GET"])],
        )
        child.state.path = path
        return child

    configs = [
        server.HttpTransportConfig(transport="sse", path="/sse"),
        server.HttpTransportConfig(transport="streamable-http", path="/mcp"),
    ]

    with patch.object(server.plex_server, "http_app", side_effect=fake_http_app):
        app = server._build_shared_http_app(configs)

    with TestClient(app, follow_redirects=False) as client:
        docs_response = client.get("/rest")
        assert docs_response.status_code == 200
        assert "Swagger UI" in docs_response.text

        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        assert "/rest/get-media" in openapi_response.json()["paths"]


def test_main_env_vars_combined_transports(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "sse,streamable-http")
    monkeypatch.setenv("MCP_HOST", "1.2.3.4")
    monkeypatch.setenv("MCP_PORT", "1234")
    monkeypatch.setenv("MCP_SSE_MOUNT", "/events")
    monkeypatch.setenv("MCP_STREAMABLE_HTTP_MOUNT", "/stream")

    with patch("mcp_plex.server.cli.uvicorn.run") as mock_run:
        server.main([])

    mock_run.assert_called_once()
    app = mock_run.call_args.args[0]
    assert mock_run.call_args.kwargs["host"] == "1.2.3.4"
    assert mock_run.call_args.kwargs["port"] == 1234
    assert [route.path for route in app.routes] == ["/events", "/stream"]


def test_main_rejects_generic_mount_for_multiple_http_transports():
    with pytest.raises(SystemExit):
        server.main(
            [
                "--transport",
                "sse",
                "--transport",
                "streamable-http",
                "--bind",
                "0.0.0.0",
                "--port",
                "8000",
                "--mount",
                "/mcp",
            ]
        )


def test_main_model_overrides():
    with patch.object(server.server, "run") as mock_run:
        server.main(
            [
                "--dense-model",
                "foo",
                "--sparse-model",
                "bar",
            ]
        )
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
        mock_run.assert_called_once_with(transport="sse", host="0.0.0.0", port=8000)


def test_env_invalid_port(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "sse")
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MCP_PORT", "not-a-port")
    with pytest.raises(SystemExit):
        server.main([])


def test_run_config_reexport():
    from mcp_plex.server import RunConfig as ExportedRunConfig

    assert ExportedRunConfig is server.RunConfig


def test_main_configures_log_level(monkeypatch):
    configured: dict[str, object] = {}

    def fake_basic_config(**kwargs):
        configured["level"] = kwargs.get("level")

    monkeypatch.setattr("logging.basicConfig", fake_basic_config)

    with patch.object(server.server, "run") as mock_run:
        server.main(["--log-level", "debug"])

    assert configured["level"] == logging.DEBUG
    mock_run.assert_called_once_with(transport="stdio")
