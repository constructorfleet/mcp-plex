"""FastMCP server exposing Plex metadata tools."""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import json
import logging
from pathlib import Path
import uuid
from typing import Annotated, Any, Callable, Mapping, Sequence, TYPE_CHECKING, cast

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastmcp.prompts import Message
from fastmcp.server import FastMCP
from fastmcp.server.context import Context as FastMCPContext
from plexapi.exceptions import PlexApiException
from plexapi.server import PlexServer as PlexServerClient
from plexapi.client import PlexClient
from plexapi.playqueue import PlayQueue
from plexapi.media import (
    AudioStream,
    MediaPartStream,
    Session as PlexSession,
    SubtitleStream,
)
from pydantic import BaseModel, Field, create_model
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from xml.etree import ElementTree

from rapidfuzz import fuzz, process
import yaml

from ..common.cache import MediaCache
from ..common.types import JSONValue
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


PlayerIdentifier = Annotated[
    str,
    Field(
        description="Friendly name, machine identifier, or client identifier of the Plex player",
        examples=["Living Room", "machine-123"],
    ),
]

MediaType = Annotated[
    str | None,
    Field(
        description="Plex media type being controlled (video/music/photo)",
        examples=["video"],
    ),
]


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
        self._watched_rating_keys: set[str] | None = None
        self._watched_rating_keys_lock = asyncio.Lock()

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
        self._watched_rating_keys = None

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

    async def get_watched_rating_keys(self) -> set[str]:
        """Return cached Plex rating keys watched by the configured user."""

        user = self.settings.recommend_user
        if not user:
            return set()
        history_limit = max(self.settings.recommend_history_limit, 0)
        if self._watched_rating_keys is not None:
            return set(self._watched_rating_keys)

        async with self._watched_rating_keys_lock:
            if self._watched_rating_keys is not None:
                return set(self._watched_rating_keys)

            try:
                plex_client = await _get_plex_client()
            except Exception as exc:  # noqa: BLE001 - allow logged fallback
                logger.warning(
                    "Unable to load Plex client for watch history: %s", exc, exc_info=exc
                )
                self._watched_rating_keys = set()
                return set()

            def _load_history() -> set[str]:
                if history_limit == 0:
                    return set()
                try:
                    account = plex_client.myPlexAccount()
                except PlexApiException as exc:  # pragma: no cover - network failure
                    logger.warning(
                        "Failed to load Plex account for watch history: %s", exc, exc_info=exc
                    )
                    return set()
                except Exception as exc:  # noqa: BLE001 - unexpected library error
                    logger.warning(
                        "Unexpected error loading Plex account: %s", exc, exc_info=exc
                    )
                    return set()

                if account is None:
                    return set()

                try:
                    plex_user = account.user(user)
                except PlexApiException as exc:  # pragma: no cover - network failure
                    logger.warning(
                        "Failed to resolve Plex user %s: %s", user, exc, exc_info=exc
                    )
                    return set()
                except Exception as exc:  # noqa: BLE001 - unexpected library error
                    logger.warning(
                        "Unexpected error resolving Plex user %s: %s", user, exc, exc_info=exc
                    )
                    return set()

                if plex_user is None:
                    return set()

                try:
                    history_kwargs: dict[str, Any] = {"server": plex_client}
                    if history_limit > 0:
                        history_kwargs["maxresults"] = history_limit
                    history_items = plex_user.history(**history_kwargs)
                except TypeError:
                    try:
                        user_id = getattr(plex_user, "id", None)
                        history_kwargs = {"accountID": user_id}
                        if history_limit > 0:
                            history_kwargs["maxresults"] = history_limit
                        history_items = plex_client.history(**history_kwargs)
                    except Exception as exc:  # noqa: BLE001 - unexpected signature change
                        logger.warning(
                            "Unable to load Plex history for %s: %s",
                            user,
                            exc,
                            exc_info=exc,
                        )
                        return set()
                except PlexApiException as exc:  # pragma: no cover - network failure
                    logger.warning(
                        "Failed to load Plex watch history for %s: %s",
                        user,
                        exc,
                        exc_info=exc,
                    )
                    return set()
                except Exception as exc:  # noqa: BLE001 - unexpected library error
                    logger.warning(
                        "Unexpected error loading Plex history for %s: %s",
                        user,
                        exc,
                        exc_info=exc,
                    )
                    return set()

                rating_keys: set[str] = set()
                for item in history_items or []:
                    rating_key = getattr(item, "ratingKey", None)
                    if rating_key is None:
                        rating_key = getattr(item, "rating_key", None)
                    normalized = media_helpers._normalize_history_rating_key(
                        rating_key
                    )
                    if normalized:
                        rating_keys.add(normalized)
                    if history_limit > 0 and len(rating_keys) >= history_limit:
                        break
                return rating_keys

            watched = await asyncio.to_thread(_load_history)
            self._watched_rating_keys = watched
            return set(watched)

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
    "recommend_media_like": "recommend-media-like",
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


def _load_configured_plex_clients() -> list[PlexClient] | None:
    """Return Plex clients defined via the configured fixture file."""

    path_value = server.settings.plex_clients_file
    if not path_value:
        return None
    path = Path(path_value)
    try:
        data = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - executed via tests
        raise RuntimeError(
            f"Configured Plex clients file '{path}' does not exist"
        ) from exc
    except OSError as exc:  # pragma: no cover - executed via tests
        raise RuntimeError(
            f"Failed to read configured Plex clients file '{path}'"
        ) from exc

    try:
        raw_entries = _parse_configured_clients_payload(path, data)
        return [_build_configured_client(entry) for entry in raw_entries]
    except Exception as exc:  # pragma: no cover - surfaced in tests
        raise RuntimeError("Failed to parse configured Plex clients file") from exc


def _configured_client_lookup() -> dict[str, PlexClient]:
    """Return configured Plex clients indexed by normalized identifiers."""

    clients = _load_configured_plex_clients()
    if not clients:
        return {}

    lookup: dict[str, PlexClient] = {}
    for client in clients:
        machine_identifier = _normalize_session_identifier(
            getattr(client, "machineIdentifier", None)
        )
        client_identifier = _normalize_session_identifier(
            getattr(client, "clientIdentifier", None)
        )
        identifier = _normalize_session_identifier(getattr(client, "identifier", None))

        for candidate in (machine_identifier, client_identifier, identifier):
            if candidate and candidate not in lookup:
                lookup[candidate] = client

        if machine_identifier and client_identifier:
            combined = f"{machine_identifier}:{client_identifier}"
            if combined not in lookup:
                lookup[combined] = client

    return lookup


def _get_configured_client(identifier: str | None) -> PlexClient | None:
    """Return the configured Plex client matching the provided identifier."""

    normalized = _normalize_session_identifier(identifier)
    if not normalized:
        return None
    lookup = _configured_client_lookup()
    if not lookup:
        return None
    return lookup.get(normalized)


def _parse_configured_clients_payload(
    path: Path, payload: str
) -> list[Mapping[str, Any]]:
    """Parse fixture payload into mapping entries."""

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        loaded = yaml.safe_load(payload) or {}
    elif suffix == ".json":
        loaded = json.loads(payload)
    else:
        return _parse_configured_clients_xml(payload)
    return _normalize_configured_clients_data(loaded)


def _parse_configured_clients_xml(payload: str) -> list[Mapping[str, Any]]:
    """Return client entries from an XML payload."""

    root = ElementTree.fromstring(payload)
    servers: list[Mapping[str, Any]] = []
    for server_element in root.findall(".//Server"):
        entry: dict[str, Any] = dict(server_element.attrib)
        xml_aliases = [
            alias_element.text
            for alias_element in server_element.findall("Alias")
        ]
        aliases = _normalize_alias_values(xml_aliases)
        if aliases:
            entry["aliases"] = aliases
        servers.append(entry)
    return servers


def _normalize_configured_clients_data(
    payload: Any,
) -> list[Mapping[str, Any]]:
    """Normalize JSON/YAML payloads into mapping entries."""

    if isinstance(payload, Mapping):
        if "MediaContainer" in payload:
            return _normalize_configured_clients_data(payload["MediaContainer"])
        if "Server" in payload:
            return _normalize_configured_clients_data(payload["Server"])
        return [value for value in payload.values() if isinstance(value, Mapping)]
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        normalized: list[Mapping[str, Any]] = []
        for item in payload:
            if isinstance(item, Mapping):
                normalized.append(item)
        return normalized
    return []


def _build_configured_client(entry: Mapping[str, Any]) -> PlexClient:
    """Construct a Plex client instance from fixture data."""

    def _normalize_value(key: str) -> str | None:
        value = entry.get(key)
        if value is None:
            return None
        value_str = str(value).strip()
        return value_str or None

    machine_identifier = _normalize_value("machineIdentifier")
    client_identifier = _normalize_value("clientIdentifier")
    identifier = (
        _normalize_value("identifier")
        or client_identifier
        or machine_identifier
    )
    host_value = _normalize_value("host")
    address = _normalize_value("address") or host_value
    scheme = _normalize_value("scheme") or "http"
    port_value = entry.get("port") or entry.get("portNumber")
    port: int | None
    if isinstance(port_value, int):
        port = port_value
    else:
        try:
            port = int(str(port_value).strip()) if port_value is not None else None
        except ValueError:
            port = None
    baseurl = _normalize_value("baseurl")
    if not baseurl and address:
        address_part = address
        if port is not None:
            address_part = f"{address_part}:{port}"
        baseurl = f"{scheme}://{address_part}"

    token = _normalize_value("token") or server.settings.plex_token
    plex_client = PlexClient(
        baseurl=baseurl,
        identifier=identifier,
        token=token,
        connect=False,
    )

    def _assign(attr: str, value: Any) -> None:
        if value is not None:
            setattr(plex_client, attr, value)

    _assign("machineIdentifier", machine_identifier)
    _assign("clientIdentifier", client_identifier)
    _assign("address", address)
    _assign("host", host_value)
    _assign("port", port)
    _assign("protocol", _normalize_value("protocol"))
    _assign("protocolVersion", _normalize_value("protocolVersion"))
    _assign("product", _normalize_value("product"))
    _assign("deviceClass", _normalize_value("deviceClass"))
    title = _normalize_value("title") or _normalize_value("name")
    _assign("title", title)
    _assign("name", title)
    provides = entry.get("protocolCapabilities") or entry.get("provides")
    if isinstance(provides, (list, tuple, set)):
        provides_str = ",".join(
            str(capability).strip()
            for capability in provides
            if str(capability).strip()
        )
    else:
        provides_str = str(provides).strip() if provides else ""
    _assign("provides", provides_str)

    aliases = _normalize_alias_values(
        entry.get("aliases") or entry.get("Alias") or entry.get("alias")
    )
    if aliases:
        setattr(plex_client, "aliases", tuple(aliases))

    return plex_client


def _normalize_alias_values(raw_aliases: Any) -> list[str]:
    """Return a list of normalized alias strings."""

    if raw_aliases is None:
        return []
    if isinstance(raw_aliases, str):
        alias_value = raw_aliases.strip()
        return [alias_value] if alias_value else []
    if isinstance(raw_aliases, Sequence) and not isinstance(
        raw_aliases, (str, bytes, bytearray)
    ):
        normalized: list[str] = []
        for alias in raw_aliases:
            if alias is None:
                continue
            alias_text = str(alias).strip()
            if alias_text and alias_text not in normalized:
                normalized.append(alias_text)
        return normalized
    return []


async def _get_plex_players() -> list[PlexPlayerMetadata]:
    """Return Plex players available for playback commands."""

    configured_clients = _load_configured_plex_clients()
    if configured_clients is not None:
        raw_clients = configured_clients
    else:
        plex_client = await _get_plex_client()

        def _load_clients() -> list[Any]:
            return list(plex_client.clients())

        raw_clients = await asyncio.to_thread(_load_clients)
    aliases: PlexPlayerAliasMap = server.settings.plex_player_aliases
    reverse_aliases: dict[str, list[str]] = {}
    for alias_key, alias_values in aliases.items():
        for alias_value in alias_values:
            if not alias_value:
                continue
            alias_list = reverse_aliases.setdefault(alias_value, [])
            if alias_key not in alias_list:
                alias_list.append(alias_key)
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

        def _add_alias(value: str | None) -> None:
            if value and value not in friendly_names:
                friendly_names.append(value)

        def _collect_alias(identifier: str | None) -> None:
            if not identifier:
                return
            for alias in aliases.get(identifier, []):
                _add_alias(alias)
            for alias in reverse_aliases.get(identifier, []):
                _add_alias(alias)

        for alias in getattr(client, "aliases", ()):  # type: ignore[arg-type]
            if alias is None:
                continue
            alias_str = str(alias).strip()
            if alias_str:
                _add_alias(alias_str)

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

    def _build_alias_graph() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
        graph: dict[str, set[str]] = {}
        labels: dict[str, set[str]] = {}
        aliases = server.settings.plex_player_aliases
        if not aliases:
            return graph, labels

        def _register(label: str) -> str | None:
            trimmed = label.strip()
            if not trimmed:
                return None
            normalized_label = trimmed.lower()
            labels.setdefault(normalized_label, set()).add(trimmed)
            graph.setdefault(normalized_label, set())
            return normalized_label

        for key, values in aliases.items():
            normalized_key = _register(key)
            if normalized_key is None:
                continue
            for value in values:
                normalized_value = _register(value)
                if normalized_value is None:
                    continue
                graph[normalized_key].add(normalized_value)
                graph[normalized_value].add(normalized_key)

        return graph, labels

    alias_graph, alias_labels = _build_alias_graph()
    alias_graph_keys = tuple(alias_graph)

    def _expand_aliases(value: str) -> set[str]:
        if not alias_graph_keys:
            return set()
        normalized_value = value.strip().lower()
        if not normalized_value:
            return set()
        target_value = normalized_value
        if target_value not in alias_graph:
            match = process.extractOne(
                normalized_value,
                alias_graph_keys,
                scorer=fuzz.WRatio,
                score_cutoff=_FUZZY_MATCH_THRESHOLD,
            )
            if match is None:
                return set()
            target_value = match[0]
        seen = {target_value}
        stack = [target_value]
        related: set[str] = set()
        while stack:
            current = stack.pop()
            for neighbor in alias_graph.get(current, set()):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
                related.update(alias_labels.get(neighbor, set()))
        return related

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
        expanded_candidates: set[str] = set()
        for candidate in candidate_strings:
            if not candidate:
                continue
            candidate_str = str(candidate).strip()
            if not candidate_str:
                continue
            expanded_candidates.add(candidate_str)
            expanded_candidates.update(_expand_aliases(candidate_str))
        for candidate in expanded_candidates:
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


async def _resolve_player_entry(query: str) -> PlexPlayerMetadata:
    """Return the Plex player metadata for the provided query string."""

    players = await _get_plex_players()
    return _match_player(query, players)


def _ensure_player_client(player: PlexPlayerMetadata) -> PlexClient:
    """Return the Plex client object for the resolved player."""

    plex_client = player.get("client")
    if plex_client is None:
        display_name = player.get("display_name") or player.get("name") or "player"
        raise ValueError(
            f"Player '{display_name}' is missing a Plex client instance"
        )
    return plex_client


async def _invoke_player_method(
    player_query: str,
    method_name: str,
    *,
    args: Sequence[Any] = (),
    media_type: str | None = "video",
) -> PlexPlayerMetadata:
    """Execute a Plex client method and return the resolved player metadata."""

    player = await _resolve_player_entry(player_query)
    plex_client = _ensure_player_client(player)
    method = getattr(plex_client, method_name, None)
    if method is None:
        display_name = player.get("display_name") or player.get("name") or "player"
        raise RuntimeError(
            f"Player '{display_name}' does not support {method_name}"
        )
    kwargs: dict[str, Any] = {}
    if media_type is not None:
        kwargs["mtype"] = media_type

    def _call() -> None:
        method(*args, **kwargs)

    try:
        await asyncio.to_thread(_call)
    except PlexApiException as exc:
        raise RuntimeError(
            f"Failed to execute {method_name} via plexapi"
        ) from exc
    return player


async def _resolve_player_timeline(player: PlexPlayerMetadata) -> Any:
    """Return the active timeline for the provided Plex player."""

    plex_client = _ensure_player_client(player)

    def _load_timeline() -> Any:
        try:
            plex_client.timelines()
        except PlexApiException as exc:
            raise RuntimeError("Failed to retrieve Plex player timeline") from exc
        return plex_client.timeline

    timeline = await asyncio.to_thread(_load_timeline)
    if timeline is None:
        display_name = player.get("display_name") or player.get("name") or "player"
        raise RuntimeError(f"Player '{display_name}' is not reporting an active timeline")
    return timeline


def _resolve_rating_key(
    media: Mapping[str, Any]
) -> tuple[str, dict[str, JSONValue]]:
    """Return the normalized rating key and Plex metadata for *media*."""

    plex_info = media_helpers._extract_plex_metadata(media)
    rating_key_value = plex_info.get("rating_key")
    rating_key_normalized = media_helpers._normalize_identifier(rating_key_value)
    if not rating_key_normalized:
        raise ValueError("Media item is missing a Plex rating key")
    return rating_key_normalized, plex_info


async def _start_playback(
    rating_key: str, player: PlexPlayerMetadata, offset_seconds: int
) -> None:
    """Send a playback command to the selected player."""

    display_name = player.get("display_name")
    plex_client = player.get("client")
    if plex_client is None:
        raise ValueError(
            f"Player '{display_name}' is missing a Plex client instance"
        )

    plex_server = await _get_plex_client()
    identity = await _fetch_plex_identity()
    offset_ms = max(offset_seconds, 0) * 1000
    plex_client_any = cast(Any, plex_client)

    def _play() -> None:
        media = plex_server.fetchItem(f"/library/metadata/{rating_key}")
        plex_client_any.playMedia(
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
    player: PlayerIdentifier,
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
    rating_key_normalized, plex_info = _resolve_rating_key(media)

    players = await _get_plex_players()
    target = _match_player(player, players)
    await _start_playback(rating_key_normalized, target, offset_seconds or 0)

    capabilities = sorted(target.get("provides", set()))

    return {
        "player": target.get("display_name"),
        "rating_key": rating_key_normalized,
        "title": plex_info.get("title") or media.get("title"),
        "offset_seconds": offset_seconds or 0,
        "player_capabilities": capabilities,
    }


@server.tool("queue-media")
async def queue_media(
    identifier: Annotated[
        str,
        Field(
            description="Rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ],
    player: PlayerIdentifier,
    play_next: Annotated[
        bool,
        Field(
            description=(
                "Insert the media immediately after the current item when true; "
                "otherwise append it to the end of the queue."
            ),
            examples=[True],
        ),
    ] = False,
) -> dict[str, Any]:
    """Add a media item to the active play queue for a Plex player."""

    media = await media_helpers._get_media_data(server, identifier)
    rating_key_normalized, plex_info = _resolve_rating_key(media)

    players = await _get_plex_players()
    target = _match_player(player, players)
    timeline = await _resolve_player_timeline(target)
    play_queue_id = getattr(timeline, "playQueueID", None)
    if not play_queue_id:
        display_name = target.get("display_name") or target.get("name") or "player"
        raise RuntimeError(f"Player '{display_name}' does not have an active play queue")

    plex_server = await _get_plex_client()

    def _enqueue() -> tuple[int, int]:
        queue = PlayQueue.get(plex_server, play_queue_id, own=True)
        media_item = plex_server.fetchItem(f"/library/metadata/{rating_key_normalized}")
        updated = queue.addItem(media_item, playNext=play_next)
        return updated.playQueueTotalCount, updated.playQueueVersion

    queue_size, queue_version = await asyncio.to_thread(_enqueue)

    return {
        "player": target.get("display_name"),
        "rating_key": rating_key_normalized,
        "title": plex_info.get("title") or media.get("title"),
        "position": "next" if play_next else "end",
        "queue_size": queue_size,
        "queue_version": queue_version,
    }


def _player_response(
    player: PlexPlayerMetadata, *, command: str, media_type: str | None
) -> dict[str, Any]:
    capabilities = sorted(player.get("provides", set()))
    return {
        "player": player.get("display_name"),
        "command": command,
        "media_type": media_type,
        "player_capabilities": capabilities,
    }


@server.tool("pause-media")
async def pause_media(
    player: PlayerIdentifier,
    media_type: MediaType = "video",
) -> dict[str, Any]:
    """Pause playback on the selected Plex player."""

    player_entry = await _invoke_player_method(player, "pause", media_type=media_type)
    return _player_response(player_entry, command="pause", media_type=media_type)


@server.tool("resume-media")
async def resume_media(
    player: PlayerIdentifier,
    media_type: MediaType = "video",
) -> dict[str, Any]:
    """Resume playback on the selected Plex player."""

    player_entry = await _invoke_player_method(player, "play", media_type=media_type)
    return _player_response(player_entry, command="resume", media_type=media_type)


@server.tool("next-media")
async def next_media(
    player: PlayerIdentifier,
    media_type: MediaType = "video",
) -> dict[str, Any]:
    """Skip to the next item on the selected Plex player."""

    player_entry = await _invoke_player_method(player, "skipNext", media_type=media_type)
    return _player_response(player_entry, command="next", media_type=media_type)


@server.tool("previous-media")
async def previous_media(
    player: PlayerIdentifier,
    media_type: MediaType = "video",
) -> dict[str, Any]:
    """Skip to the previous item on the selected Plex player."""

    player_entry = await _invoke_player_method(
        player, "skipPrevious", media_type=media_type
    )
    return _player_response(player_entry, command="previous", media_type=media_type)


@server.tool("fastforward-media")
async def fastforward_media(
    player: PlayerIdentifier,
    media_type: MediaType = "video",
) -> dict[str, Any]:
    """Step forward in the current item on the selected Plex player."""

    player_entry = await _invoke_player_method(
        player, "stepForward", media_type=media_type
    )
    return _player_response(
        player_entry, command="fastforward", media_type=media_type
    )


@server.tool("rewind-media")
async def rewind_media(
    player: PlayerIdentifier,
    media_type: MediaType = "video",
) -> dict[str, Any]:
    """Step backward in the current item on the selected Plex player."""

    player_entry = await _invoke_player_method(player, "stepBack", media_type=media_type)
    return _player_response(player_entry, command="rewind", media_type=media_type)


@server.tool("set-subtitle")
async def set_subtitle(
    player: PlayerIdentifier,
    subtitle_language: Annotated[
        str,
        Field(
            description="Subtitle language code from the media metadata",
            examples=["spa"],
        ),
    ],
    media_type: MediaType = "video",
) -> dict[str, Any]:
    """Select the subtitle language for the current playback session."""

    if not subtitle_language:
        raise ValueError("subtitle language is required")
    player_entry = await _resolve_player_entry(player)
    stream = await _resolve_subtitle_stream(player_entry, subtitle_language)
    plex_client = _ensure_player_client(player_entry)
    method = getattr(plex_client, "setSubtitleStream", None)
    if method is None:
        display_name = (
            player_entry.get("display_name")
            or player_entry.get("name")
            or "player"
        )
        raise RuntimeError(f"Player '{display_name}' does not support setSubtitleStream")
    kwargs: dict[str, Any] = {}
    if media_type is not None:
        kwargs["mtype"] = media_type

    def _call() -> None:
        target = stream if stream is not None else subtitle_language
        method(target, **kwargs)

    try:
        await asyncio.to_thread(_call)
    except PlexApiException as exc:
        raise RuntimeError("Failed to execute setSubtitleStream via plexapi") from exc
    response = _player_response(player_entry, command="set-subtitle", media_type=media_type)
    response["subtitle_language"] = subtitle_language
    if stream is not None:
        stream_id = getattr(stream, "id", None)
        if stream_id is not None:
            response["subtitle_stream_id"] = stream_id
    return response


def _normalize_session_identifier(value: Any) -> str:
    if value is None:
        return ""
    try:
        normalized = str(value)
    except Exception:
        return ""
    return normalized.strip().lower()


def _collect_session_players(session: PlexSession | Any) -> list[Any]:
    players = getattr(session, "players", None)
    if isinstance(players, (list, tuple, set)):
        return [player for player in players if player is not None]
    if players is not None:
        try:
            return [player for player in players if player is not None]
        except TypeError:
            pass
    player = getattr(session, "player", None)
    if player is None:
        return []
    if isinstance(player, str):
        configured = _get_configured_client(player)
        if configured is not None:
            return [configured]
    return [player]


def _collect_audio_streams(session: PlexSession | Any) -> list[AudioStream]:
    streams: list[AudioStream] = []
    getter = getattr(session, "audioStreams", None)
    if callable(getter):
        try:
            result = getter()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to fetch audio streams: %s", exc, exc_info=exc)
        else:
            if isinstance(result, (list, tuple, set)):
                streams.extend(cast(Sequence[AudioStream], result))
            elif result is not None:
                streams.append(cast(AudioStream, result))
    return streams


def _collect_subtitle_streams(session: PlexSession | Any) -> list[SubtitleStream]:
    streams: list[SubtitleStream] = []
    getter = getattr(session, "subtitleStreams", None)
    if callable(getter):
        try:
            result = getter()
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Failed to fetch subtitle streams: %s", exc, exc_info=exc)
        else:
            if isinstance(result, (list, tuple, set)):
                streams.extend(cast(Sequence[SubtitleStream], result))
            elif result is not None:
                streams.append(cast(SubtitleStream, result))
    if not streams:
        media_items = getattr(session, "media", None)
        if isinstance(media_items, (list, tuple, set)):
            for media in media_items:
                parts = getattr(media, "parts", None)
                if not isinstance(parts, (list, tuple, set)):
                    continue
                for part in parts:
                    part_streams = getattr(part, "streams", None)
                    if not isinstance(part_streams, (list, tuple, set)):
                        continue
                    for stream in part_streams:
                        stream_type = _coerce_int(getattr(stream, "streamType", None))
                        if stream_type == 3:
                            streams.append(cast(SubtitleStream, stream))
    return streams


async def _find_player_session(player: PlexPlayerMetadata) -> PlexSession | None:
    machine_identifier = _normalize_session_identifier(
        player.get("machine_identifier")
    )
    client_identifier = _normalize_session_identifier(
        player.get("client_identifier")
    )
    if not machine_identifier and not client_identifier:
        return None
    plex_client = await _get_plex_client()

    def _load_sessions() -> list[PlexSession]:
        sessions = plex_client.sessions()
        if isinstance(sessions, list):
            return cast(list[PlexSession], sessions)
        try:
            return list(cast(Sequence[PlexSession], sessions))
        except TypeError:
            return [cast(PlexSession, sessions)]

    try:
        sessions = await asyncio.to_thread(_load_sessions)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.debug("Failed to load Plex sessions: %s", exc, exc_info=exc)
        return None

    for session in sessions:
        for candidate in _collect_session_players(session):
            session_machine = _normalize_session_identifier(
                getattr(candidate, "machineIdentifier", None)
                or getattr(candidate, "machine_identifier", None)
            )
            session_client = _normalize_session_identifier(
                getattr(candidate, "clientIdentifier", None)
                or getattr(candidate, "client_identifier", None)
            )
            if machine_identifier and session_machine == machine_identifier:
                return session
            if client_identifier and session_client == client_identifier:
                return session
    return None


def _normalize_stream_language(stream: MediaPartStream | Any) -> str | None:
    for attribute in ("languageTag", "languageCode", "language", "locale"):
        value = getattr(stream, attribute, None)
        if value is None:
            continue
        try:
            text = str(value)
        except Exception:
            continue
        normalized = text.strip().lower()
        if not normalized:
            continue
        if "-" in normalized:
            normalized = normalized.split("-", 1)[0]
        return normalized
    return None


def _coerce_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, bool):
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def _extract_audio_channel_count(stream: AudioStream | MediaPartStream | Any) -> int:
    channels = _coerce_int(getattr(stream, "channels", None))
    if channels:
        return channels
    return _coerce_int(getattr(stream, "audioChannelCount", None))


def _audio_stream_rank(
    stream: AudioStream | MediaPartStream | Any,
) -> tuple[int, int, int]:
    channels = _extract_audio_channel_count(stream)
    bitrate = _coerce_int(getattr(stream, "bitrate", None))
    stream_id = _coerce_int(getattr(stream, "id", None))
    return (channels, bitrate, stream_id)


def _select_best_audio_stream(
    session: PlexSession | Any, audio_language: str
) -> AudioStream | None:
    normalized_language = audio_language.strip().lower()
    if not normalized_language:
        return None
    streams = _collect_audio_streams(session)
    if not streams:
        return None
    best_stream: AudioStream | None = None
    best_rank: tuple[int, int, int] | None = None
    for stream in streams:
        stream_language = _normalize_stream_language(stream)
        if stream_language != normalized_language:
            continue
        rank = _audio_stream_rank(stream)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_stream = stream
    return best_stream


def _select_subtitle_stream(
    session: PlexSession | Any, subtitle_language: str
) -> SubtitleStream | None:
    normalized_language = subtitle_language.strip().lower()
    if not normalized_language:
        return None
    streams = _collect_subtitle_streams(session)
    if not streams:
        return None
    for stream in streams:
        stream_language = _normalize_stream_language(stream)
        if stream_language == normalized_language:
            return stream
    return None


async def _resolve_audio_stream(
    player: PlexPlayerMetadata, audio_language: str
) -> AudioStream | None:
    session = await _find_player_session(player)
    if session is None:
        return None
    return _select_best_audio_stream(session, audio_language)


async def _resolve_subtitle_stream(
    player: PlexPlayerMetadata, subtitle_language: str
) -> SubtitleStream | None:
    session = await _find_player_session(player)
    if session is None:
        return None
    return _select_subtitle_stream(session, subtitle_language)


@server.tool("set-audio")
async def set_audio(
    player: PlayerIdentifier,
    audio_language: Annotated[
        str,
        Field(
            description="Audio language code from the media metadata",
            examples=["eng"],
        ),
    ],
    media_type: MediaType = "video",
) -> dict[str, Any]:
    """Select the audio language for the current playback session."""

    if not audio_language:
        raise ValueError("audio language is required")
    player_entry = await _resolve_player_entry(player)
    stream = await _resolve_audio_stream(player_entry, audio_language)
    plex_client = _ensure_player_client(player_entry)
    method = getattr(plex_client, "setAudioStream", None)
    if method is None:
        display_name = (
            player_entry.get("display_name")
            or player_entry.get("name")
            or "player"
        )
        raise RuntimeError(f"Player '{display_name}' does not support setAudioStream")
    kwargs: dict[str, Any] = {}
    if media_type is not None:
        kwargs["mtype"] = media_type

    def _call() -> None:
        target = stream if stream is not None else audio_language
        method(target, **kwargs)

    try:
        await asyncio.to_thread(_call)
    except PlexApiException as exc:
        raise RuntimeError("Failed to execute setAudioStream via plexapi") from exc
    response = _player_response(player_entry, command="set-audio", media_type=media_type)
    response["audio_language"] = audio_language
    if stream is not None:
        channels = _extract_audio_channel_count(stream)
        if channels:
            response["audio_channels"] = channels
        stream_id = getattr(stream, "id", None)
        if stream_id is not None:
            response["audio_stream_id"] = stream_id
    return response


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
