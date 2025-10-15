"""FastMCP server exposing Plex metadata tools."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import json
import logging
import uuid
from typing import Annotated, Any, Callable, Sequence, TYPE_CHECKING, cast

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastmcp.prompts import Message
from fastmcp.server import FastMCP
from fastmcp.server.context import Context as FastMCPContext
from plexapi.exceptions import PlexApiException
from plexapi.server import PlexServer as PlexServerClient
from plexapi.client import PlexClient
from pydantic import BaseModel, Field, create_model
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from rapidfuzz import fuzz, process

from ..common.cache import MediaCache
from . import media as media_helpers
from .config import PlexPlayerAliasMap, Settings
from .models import (
    PlexPlayerMetadata,
)
from .tools.media_library import register_media_library_tools


logger = logging.getLogger(__name__)

try:
    from sentence_transformers import CrossEncoder
except Exception as exc:
    logger.warning(
        "Failed to import CrossEncoder for reranking: %s",
        exc,
        exc_info=exc,
    )
    CrossEncoder = None


settings = Settings()
SERVER_NAME = "Plex Media"


try:
    __version__ = importlib.metadata.version("mcp-plex")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"


class PlexServer(FastMCP):
    """FastMCP server with an attached Qdrant client."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        qdrant_client: AsyncQdrantClient | None = None,
    ) -> None:  # noqa: D401 - short description inherited
        self._settings = settings or Settings()
        location = self.settings.qdrant_url
        host = self.settings.qdrant_host
        if location is None and host is None:
            location = ":memory:"
        self._qdrant_config = {
            "location": location,
            "api_key": self.settings.qdrant_api_key,
            "host": host,
            "port": self.settings.qdrant_port,
            "grpc_port": self.settings.qdrant_grpc_port,
            "prefer_grpc": self.settings.qdrant_prefer_grpc,
            "https": self.settings.qdrant_https,
        }
        self._qdrant_client_factory = None
        if qdrant_client is None:
            self._qdrant_client_factory = self._build_default_qdrant_client
            self._qdrant_client = self._build_default_qdrant_client()
        else:
            self._qdrant_client = qdrant_client

        class _ServerLifespan:
            def __init__(self, plex_server: "PlexServer") -> None:
                self._plex_server = plex_server

            async def __aenter__(self) -> None:  # noqa: D401 - matching protocol
                return None

            async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
                await self._plex_server.close()

        def _lifespan(app: FastMCP) -> _ServerLifespan:  # noqa: ARG001
            return _ServerLifespan(self)

        super().__init__(name=SERVER_NAME, lifespan=_lifespan)
        self._reranker: CrossEncoder | None = None
        self._reranker_loaded = False
        self._reranker_lock = asyncio.Lock()
        self._ensure_reranker_task: asyncio.Task[CrossEncoder | None] | None = None
        self.cache = MediaCache(self.settings.cache_size)
        self.client_identifier = uuid.uuid4().hex
        self._plex_identity: dict[str, Any] | None = None
        self._plex_client: PlexServerClient | None = None
        self._plex_client_lock = asyncio.Lock()

    def _build_default_qdrant_client(self) -> AsyncQdrantClient:
        """Construct a new Qdrant client using the server settings."""

        return AsyncQdrantClient(**self._qdrant_config)

    @property
    def qdrant_client(self) -> AsyncQdrantClient:
        if self._qdrant_client is None:
            if self._qdrant_client_factory is None:
                raise RuntimeError("Qdrant client is not configured")
            self._qdrant_client = self._qdrant_client_factory()
        return self._qdrant_client

    @qdrant_client.setter
    def qdrant_client(self, client: AsyncQdrantClient | None) -> None:
        self._qdrant_client = client
        if client is None:
            return
        self._qdrant_client_factory = None

    async def close(self) -> None:
        if self._qdrant_client is not None:
            await self._qdrant_client.close()
        if self._qdrant_client_factory is not None:
            self._qdrant_client = None
        self._plex_client = None
        self._plex_identity = None

    @property
    def settings(self) -> Settings:  # type: ignore[override]
        return self._settings

    @property
    def reranker(self) -> CrossEncoder | None:
        if not self.settings.use_reranker or CrossEncoder is None:
            return None
        if self._reranker_loaded:
            return self._reranker
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ensure_reranker())
        else:
            if self._ensure_reranker_task is None or self._ensure_reranker_task.done():
                self._ensure_reranker_task = loop.create_task(self.ensure_reranker())
            return self._reranker

    async def ensure_reranker(self) -> CrossEncoder | None:
        if not self.settings.use_reranker or CrossEncoder is None:
            self._reranker_loaded = True
            self._reranker = None
            return None
        if self._reranker_loaded:
            return self._reranker
        async with self._reranker_lock:
            if self._reranker_loaded:
                return self._reranker
            try:
                reranker = await asyncio.to_thread(
                    CrossEncoder, self.settings.reranker_model
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize CrossEncoder reranker: %s",
                    exc,
                    exc_info=exc,
                )
                reranker = None
            self._reranker = reranker
            self._reranker_loaded = True
        return self._reranker

    def clear_plex_identity_cache(self) -> None:
        """Reset cached Plex identity metadata."""

        self._plex_identity = None
        self._plex_client = None


server = PlexServer(settings=settings)
register_media_library_tools(server)

_MEDIA_TOOL_EXPORTS = {
    "get_media": "get-media",
    "search_media": "search-media",
    "query_media": "query-media",
    "recommend_media": "recommend-media",
    "new_movies": "new-movies",
    "new_shows": "new-shows",
    "actor_movies": "actor-movies",
}

_MEDIA_RESOURCE_EXPORTS = {
    "media_item": "resource://media-item/{identifier}",
    "media_ids": "resource://media-ids/{identifier}",
    "media_poster": "resource://media-poster/{identifier}",
    "media_background": "resource://media-background/{identifier}",
}

_MEDIA_PROMPT_EXPORTS = {"media_info": "media-info"}

for attr_name, tool_name in _MEDIA_TOOL_EXPORTS.items():
    globals()[attr_name] = server._tool_manager._tools[tool_name]

for attr_name, resource_uri in _MEDIA_RESOURCE_EXPORTS.items():
    globals()[attr_name] = server._resource_manager._templates[resource_uri]

for attr_name, prompt_name in _MEDIA_PROMPT_EXPORTS.items():
    globals()[attr_name] = server._prompt_manager._prompts[prompt_name]


def _request_model(name: str, fn: Callable[..., object]) -> type[BaseModel] | None:
    """Generate a Pydantic model representing the callable's parameters."""

    signature = inspect.signature(fn)
    if not signature.parameters:
        return None

    fields: dict[str, tuple[object, object]] = {}
    for param_name, parameter in signature.parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue

        annotation: object
        if parameter.annotation is inspect.Signature.empty:
            annotation = object
        else:
            annotation = parameter.annotation

        default: object
        if parameter.default is inspect.Signature.empty:
            default = ...
        else:
            default = parameter.default

        fields[param_name] = (annotation, default)

    if not fields:
        return None

    model_name = "".join(
        part.capitalize() for part in name.replace("-", "_").split("_")
    )
    model_name = f"{model_name or 'Request'}Request"
    request_model = create_model(model_name, **fields)  # type: ignore[arg-type]
    return request_model


def _ensure_plex_configuration() -> None:
    """Ensure Plex playback settings are provided."""

    if not server.settings.plex_url or not server.settings.plex_token:
        raise RuntimeError("PLEX_URL and PLEX_TOKEN must be configured for playback")


async def _get_plex_client() -> PlexServerClient:
    """Return a cached Plex API client instance."""

    _ensure_plex_configuration()
    async with server._plex_client_lock:
        if server._plex_client is None:
            base_url = str(server.settings.plex_url)

            def _connect() -> PlexServerClient:
                return PlexServerClient(base_url, server.settings.plex_token)

            server._plex_client = await asyncio.to_thread(_connect)
        return server._plex_client


async def _fetch_plex_identity() -> dict[str, Any]:
    """Fetch and cache Plex server identity details."""

    if server._plex_identity is not None:
        return server._plex_identity
    plex_client = await _get_plex_client()
    machine_identifier = getattr(plex_client, "machineIdentifier", None)
    if not machine_identifier:
        raise RuntimeError("Unable to determine Plex server machine identifier")
    server._plex_identity = {"machineIdentifier": machine_identifier}
    return server._plex_identity


async def _get_plex_players() -> list[PlexPlayerMetadata]:
    """Return Plex players available for playback commands."""

    plex_client = await _get_plex_client()

    def _load_clients() -> list[Any]:
        return list(plex_client.clients())

    raw_clients = await asyncio.to_thread(_load_clients)
    aliases: PlexPlayerAliasMap = server.settings.plex_player_aliases
    players: list[PlexPlayerMetadata] = []

    for client in raw_clients:
        provides_raw = getattr(client, "provides", "")
        if isinstance(provides_raw, str):
            provides_iterable = provides_raw.split(",")
        elif isinstance(provides_raw, (list, tuple, set)):
            provides_iterable = provides_raw
        else:
            provides_iterable = []
        provides = {
            str(capability).strip().lower()
            for capability in provides_iterable
            if str(capability).strip()
        }
        machine_id = getattr(client, "machineIdentifier", None)
        client_id = getattr(client, "clientIdentifier", None)
        address = getattr(client, "address", None)
        port_value = getattr(client, "port", None)
        port: int | None
        if isinstance(port_value, int):
            port = port_value
        else:
            try:
                port = int(port_value)
            except (TypeError, ValueError):
                port = None
        name = getattr(client, "title", None) or getattr(client, "name", None)
        product = getattr(client, "product", None) or getattr(client, "device", None)

        friendly_names: list[str] = []

        def _collect_alias(identifier: str | None) -> None:
            if not identifier:
                return
            for alias in aliases.get(identifier, []):
                if alias and alias not in friendly_names:
                    friendly_names.append(alias)

        _collect_alias(machine_id)
        _collect_alias(client_id)
        if machine_id and client_id:
            _collect_alias(f"{machine_id}:{client_id}")

        display_name = (
            friendly_names[0]
            if friendly_names
            else name or product or machine_id or client_id or "Unknown player"
        )

        if display_name not in friendly_names:
            friendly_names.append(display_name)

        entry: PlexPlayerMetadata = {
            "display_name": display_name,
            "friendly_names": friendly_names,
            "provides": provides,
            "client": cast(PlexClient | None, client),
        }
        if name:
            entry["name"] = str(name)
        if product:
            entry["product"] = str(product)
        if machine_id:
            entry["machine_identifier"] = str(machine_id)
        if client_id:
            entry["client_identifier"] = str(client_id)
        if address:
            entry["address"] = str(address)
        if port is not None:
            entry["port"] = port

        players.append(entry)

    return players


_FUZZY_MATCH_THRESHOLD = 70


def _match_player(
    query: str, players: Sequence[PlexPlayerMetadata]
) -> PlexPlayerMetadata:
    """Locate a Plex player by friendly name or identifier."""

    normalized_query = query.strip()
    normalized = normalized_query.lower()
    if not normalized_query:
        raise ValueError(f"Player '{query}' not found")

    candidate_entries: list[tuple[str, str, PlexPlayerMetadata]] = []
    for player in players:
        candidate_strings = {
            player.get("display_name"),
            player.get("name"),
            player.get("product"),
            player.get("machine_identifier"),
            player.get("client_identifier"),
        }
        candidate_strings.update(player.get("friendly_names", []))
        machine_id = player.get("machine_identifier")
        client_id = player.get("client_identifier")
        if machine_id and client_id:
            candidate_strings.add(f"{machine_id}:{client_id}")
        for candidate in candidate_strings:
            if not candidate:
                continue
            candidate_str = str(candidate).strip()
            if not candidate_str:
                continue
            candidate_lower = candidate_str.lower()
            candidate_entries.append((candidate_str, candidate_lower, player))
            if candidate_lower == normalized:
                return player

    def _process_choice(choice: str | tuple[str, str, dict[str, Any]]) -> str:
        if isinstance(choice, tuple):
            return choice[1]
        return str(choice).strip().lower()

    match = process.extractOne(
        normalized_query,
        candidate_entries,
        scorer=fuzz.WRatio,
        processor=_process_choice,
        score_cutoff=_FUZZY_MATCH_THRESHOLD,
    )
    if match:
        choice, _, _ = match
        if choice is not None:
            return choice[2]
    raise ValueError(f"Player '{query}' not found")


async def _start_playback(
    rating_key: str, player: PlexPlayerMetadata, offset_seconds: int
) -> None:
    """Send a playback command to the selected player."""

    provides = player.get("provides", set())
    if "player" not in provides:
        raise ValueError(
            f"Player '{player.get('display_name')}' cannot be controlled for playback"
        )
    plex_client = player.get("client")
    if plex_client is None:
        raise ValueError(
            f"Player '{player.get('display_name')}' is missing a Plex client instance"
        )

    plex_server = await _get_plex_client()
    identity = await _fetch_plex_identity()
    offset_ms = max(offset_seconds, 0) * 1000

    def _play() -> None:
        media = plex_server.fetchItem(f"/library/metadata/{rating_key}")
        cast(Any, plex_client).playMedia(
            media,
            offset=offset_ms,
            machineIdentifier=identity["machineIdentifier"],
        )

    try:
        await asyncio.to_thread(_play)
    except PlexApiException as exc:
        raise RuntimeError("Failed to start playback via plexapi") from exc


@server.tool("play-media")
async def play_media(
    identifier: Annotated[
        str,
        Field(
            description="Rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ],
    player: Annotated[
        str,
        Field(
            description=(
                "Friendly name, machine identifier, or client identifier of the"
                " Plex player"
            ),
            examples=["Living Room", "machine-123"],
        ),
    ],
    offset_seconds: Annotated[
        int | None,
        Field(
            description="Start playback at the specified offset (seconds)",
            ge=0,
            examples=[0],
        ),
    ] = 0,
) -> dict[str, Any]:
    """Play a media item on a specific Plex player."""

    media = await media_helpers._get_media_data(server, identifier)
    plex_info = media_helpers._extract_plex_metadata(media)
    rating_key_value = plex_info.get("rating_key")
    rating_key_normalized = media_helpers._normalize_identifier(rating_key_value)
    if not rating_key_normalized:
        raise ValueError("Media item is missing a Plex rating key")

    players = await _get_plex_players()
    target = _match_player(player, players)
    await _start_playback(rating_key_normalized, target, offset_seconds or 0)

    return {
        "player": target.get("display_name"),
        "rating_key": rating_key_normalized,
        "title": plex_info.get("title") or media.get("title"),
        "offset_seconds": offset_seconds or 0,
    }


@server.custom_route("/rest", methods=["GET"])
async def rest_docs(request: Request) -> Response:
    """Serve Swagger UI for REST endpoints."""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MCP REST API")


def _build_openapi_schema() -> dict[str, object]:
    app = FastAPI()
    for name, tool in server._tool_manager._tools.items():
        request_model = _request_model(name, tool.fn)

        if request_model is None:
            app.post(f"/rest/{name}")(tool.fn)
            continue

        async def _tool_stub(payload: request_model) -> None:  # type: ignore[name-defined]
            pass

        _tool_stub.__name__ = f"tool_{name.replace('-', '_')}"
        _tool_stub.__doc__ = tool.fn.__doc__
        _tool_stub.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "payload",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=request_model,
                )
            ],
            return_annotation=inspect.Signature.empty,
        )

        app.post(f"/rest/{name}")(_tool_stub)
    for name, prompt in server._prompt_manager._prompts.items():

        async def _p_stub(**kwargs):  # noqa: ARG001
            pass

        _p_stub.__name__ = f"prompt_{name.replace('-', '_')}"
        _p_stub.__doc__ = prompt.fn.__doc__
        request_model = _request_model(name, prompt.fn)
        prompt_signature = inspect.signature(prompt.fn)
        if request_model is None:
            _p_stub.__signature__ = prompt_signature.replace(
                return_annotation=inspect.Signature.empty
            )
        else:
            _p_stub.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        "payload",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=request_model,
                    )
                ],
                return_annotation=inspect.Signature.empty,
            )
        app.post(f"/rest/prompt/{name}")(_p_stub)
    for uri, resource in server._resource_manager._templates.items():
        path = uri.replace("resource://", "")

        async def _r_stub(**kwargs):  # noqa: ARG001
            pass

        _r_stub.__name__ = (
            f"resource_{path.replace('/', '_').replace('{', '').replace('}', '')}"
        )
        _r_stub.__doc__ = resource.fn.__doc__
        _r_stub.__signature__ = inspect.signature(resource.fn).replace(
            return_annotation=inspect.Signature.empty
        )
        app.get(f"/rest/resource/{path}")(_r_stub)
    return get_openapi(title="MCP REST API", version="1.0.0", routes=app.routes)


_OPENAPI_SCHEMA = _build_openapi_schema()


@server.custom_route("/openapi.json", methods=["GET"])
async def openapi_json(request: Request) -> Response:  # noqa: ARG001
    """Return the OpenAPI schema for REST endpoints."""
    return JSONResponse(_OPENAPI_SCHEMA)


def _register_rest_endpoints() -> None:
    def _register(
        path: str, method: str, handler: Callable, fn: Callable, name: str
    ) -> None:
        handler.__name__ = name
        handler.__doc__ = fn.__doc__
        handler.__signature__ = inspect.signature(fn).replace(
            return_annotation=inspect.Signature.empty
        )
        server.custom_route(path, methods=[method])(handler)

    for name, tool in server._tool_manager._tools.items():

        async def _rest_tool(request: Request, _tool=tool) -> Response:  # noqa: ARG001
            try:
                arguments = await request.json()
            except Exception:
                arguments = {}
            async with FastMCPContext(fastmcp=server):
                result = await _tool.fn(**arguments)
            return JSONResponse(result)

        _register(
            f"/rest/{name}",
            "POST",
            _rest_tool,
            tool.fn,
            f"rest_{name.replace('-', '_')}",
        )

    for name, prompt in server._prompt_manager._prompts.items():

        async def _rest_prompt(request: Request, _prompt=prompt) -> Response:  # noqa: ARG001
            try:
                arguments = await request.json()
            except Exception:
                arguments = None
            async with FastMCPContext(fastmcp=server):
                messages = await _prompt.render(arguments)
            return JSONResponse([m.model_dump() for m in messages])

        _register(
            f"/rest/prompt/{name}",
            "POST",
            _rest_prompt,
            prompt.fn,
            f"rest_prompt_{name.replace('-', '_')}",
        )

    for uri, resource in server._resource_manager._templates.items():
        path = uri.replace("resource://", "")

        async def _rest_resource(
            request: Request, _uri_template=uri, _resource=resource
        ) -> Response:
            formatted = _uri_template
            for key, value in request.path_params.items():
                formatted = formatted.replace(f"{{{key}}}", value)
            async with FastMCPContext(fastmcp=server):
                data = await server._resource_manager.read_resource(formatted)
            if isinstance(data, bytes):
                return Response(content=data, media_type=_resource.mime_type)
            try:
                return JSONResponse(json.loads(data), media_type=_resource.mime_type)
            except Exception:
                return PlainTextResponse(str(data), media_type=_resource.mime_type)

        handler_name = (
            f"rest_resource_{path.replace('/', '_').replace('{', '').replace('}', '')}"
        )
        _register(
            f"/rest/resource/{path}",
            "GET",
            _rest_resource,
            resource.fn,
            handler_name,
        )


_register_rest_endpoints()


def main(argv: list[str] | None = None) -> None:
    """Entry point retained for backwards compatibility."""

    from .cli import main as cli_main

    cli_main(argv)


if __name__ == "__main__":
    main()


if TYPE_CHECKING:
    from .cli import RunConfig as RunConfig


def __getattr__(name: str) -> Any:
    if name == "RunConfig":
        from .cli import RunConfig as _RunConfig

        return _RunConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PlexServer",
    "server",
    "settings",
    "main",
    "RunConfig",
    "Message",
]
