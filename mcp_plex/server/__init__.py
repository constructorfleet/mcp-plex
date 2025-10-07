"""FastMCP server exposing Plex metadata tools."""
from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import json
import logging
import uuid
from typing import Annotated, Any, Callable, Mapping, Sequence, TYPE_CHECKING, cast
from typing import NotRequired, TypedDict

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
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from rapidfuzz import fuzz, process

from ..common.cache import MediaCache
from ..common.types import JSONValue
from .config import PlexPlayerAliasMap, Settings


class PlexTag(TypedDict, total=False):
    """Representation of a Plex tag entry (actor, director, etc.)."""

    tag: NotRequired[str]
    name: NotRequired[str]


PersonEntry = str | PlexTag


class ExternalIds(TypedDict, total=False):
    """External identifier payload for indexed media."""

    id: NotRequired[str | int | None]


class PlexMediaMetadata(TypedDict, total=False):
    """Subset of Plex metadata stored in Qdrant payloads."""

    rating_key: NotRequired[str]
    guid: NotRequired[str]
    title: NotRequired[str]
    type: NotRequired[str]
    thumb: NotRequired[str]
    art: NotRequired[str]
    summary: NotRequired[str]
    tagline: NotRequired[str | list[str]]
    added_at: NotRequired[int]
    year: NotRequired[int]
    directors: NotRequired[list[PersonEntry]]
    writers: NotRequired[list[PersonEntry]]
    actors: NotRequired[list[PersonEntry]]
    grandparent_title: NotRequired[str]
    parent_title: NotRequired[str]
    index: NotRequired[int]
    parent_index: NotRequired[int]
    grandparent_thumb: NotRequired[str]
    original_title: NotRequired[str]


class AggregatedMediaItem(TypedDict, total=False):
    """Flattened media payload combining Plex and external data."""

    title: NotRequired[str]
    summary: NotRequired[str]
    type: NotRequired[str]
    year: NotRequired[int]
    added_at: NotRequired[int]
    show_title: NotRequired[str]
    season_number: NotRequired[int]
    episode_number: NotRequired[int]
    tagline: NotRequired[str | list[str]]
    reviews: NotRequired[list[str]]
    overview: NotRequired[str]
    plot: NotRequired[str]
    genres: NotRequired[list[str]]
    collections: NotRequired[list[str]]
    actors: NotRequired[list[PersonEntry]]
    directors: NotRequired[list[PersonEntry]]
    writers: NotRequired[list[PersonEntry]]
    imdb: NotRequired[ExternalIds]
    tmdb: NotRequired[ExternalIds]
    tvdb: NotRequired[ExternalIds]
    plex: NotRequired[PlexMediaMetadata]


class QdrantMediaPayload(TypedDict, total=False):
    """Raw payload stored within Qdrant records."""

    data: NotRequired[AggregatedMediaItem]
    title: NotRequired[str]
    summary: NotRequired[str]
    type: NotRequired[str]
    year: NotRequired[int]
    added_at: NotRequired[int]
    show_title: NotRequired[str]
    season_number: NotRequired[int]
    episode_number: NotRequired[int]
    tagline: NotRequired[str | list[str]]
    reviews: NotRequired[list[str]]
    overview: NotRequired[str]
    plot: NotRequired[str]
    genres: NotRequired[list[str]]
    collections: NotRequired[list[str]]
    actors: NotRequired[list[PersonEntry]]
    directors: NotRequired[list[PersonEntry]]
    writers: NotRequired[list[PersonEntry]]
    imdb: NotRequired[ExternalIds]
    tmdb: NotRequired[ExternalIds]
    tvdb: NotRequired[ExternalIds]
    plex: NotRequired[PlexMediaMetadata]


class PlexPlayerMetadata(TypedDict, total=False):
    """Metadata describing a Plex player that can receive playback commands."""

    name: NotRequired[str]
    product: NotRequired[str]
    display_name: str
    friendly_names: list[str]
    machine_identifier: NotRequired[str]
    client_identifier: NotRequired[str]
    address: NotRequired[str]
    port: NotRequired[int]
    provides: set[str]
    client: NotRequired[PlexClient | None]


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
        self.qdrant_client = qdrant_client or AsyncQdrantClient(
            location=location,
            api_key=self.settings.qdrant_api_key,
            host=host,
            port=self.settings.qdrant_port,
            grpc_port=self.settings.qdrant_grpc_port,
            prefer_grpc=self.settings.qdrant_prefer_grpc,
            https=self.settings.qdrant_https,
        )

        class _ServerLifespan:
            def __init__(self, plex_server: "PlexServer") -> None:
                self._plex_server = plex_server

            async def __aenter__(self) -> None:  # noqa: D401 - matching protocol
                return None

            async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
                await self._plex_server.close()

        def _lifespan(app: FastMCP) -> _ServerLifespan:  # noqa: ARG001
            return _ServerLifespan(self)

        super().__init__(lifespan=_lifespan)
        self._reranker: CrossEncoder | None = None
        self._reranker_loaded = False
        self.cache = MediaCache(self.settings.cache_size)
        self.client_identifier = uuid.uuid4().hex
        self._plex_identity: dict[str, Any] | None = None
        self._plex_client: PlexServerClient | None = None
        self._plex_client_lock = asyncio.Lock()

    async def close(self) -> None:
        await self.qdrant_client.close()
        self._plex_client = None
        self._plex_identity = None

    @property
    def settings(self) -> Settings:  # type: ignore[override]
        return self._settings

    @property
    def reranker(self) -> CrossEncoder | None:
        if not self.settings.use_reranker or CrossEncoder is None:
            return None
        if not self._reranker_loaded:
            try:
                self._reranker = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2"
                )
            except Exception as exc:
                logger.warning(
                    "Failed to initialize CrossEncoder reranker: %s",
                    exc,
                    exc_info=exc,
                )
                self._reranker = None
            self._reranker_loaded = True
        return self._reranker

    def clear_plex_identity_cache(self) -> None:
        """Reset cached Plex identity metadata."""

        self._plex_identity = None
        self._plex_client = None


server = PlexServer(settings=settings)


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

    model_name = "".join(part.capitalize() for part in name.replace("-", "_").split("_"))
    model_name = f"{model_name or 'Request'}Request"
    request_model = create_model(model_name, **fields)  # type: ignore[arg-type]
    return request_model


async def _find_records(identifier: str, limit: int = 5) -> list[models.Record]:
    """Locate records matching an identifier or title."""
    # First, try direct ID lookup
    try:
        record_id: Any = int(identifier) if identifier.isdigit() else identifier
        recs = await server.qdrant_client.retrieve(
            "media-items", ids=[record_id], with_payload=True
        )
        if recs:
            return recs
    except Exception:
        pass

    should = [
        models.FieldCondition(key="data.plex.rating_key", match=models.MatchValue(value=identifier)),
        models.FieldCondition(key="data.imdb.id", match=models.MatchValue(value=identifier)),
    ]
    # TMDb ids are integers
    if identifier.isdigit():
        should.append(
            models.FieldCondition(
                key="data.tmdb.id", match=models.MatchValue(value=int(identifier))
            )
        )
    should.append(
        models.FieldCondition(key="title", match=models.MatchText(text=identifier))
    )
    flt = models.Filter(should=should)
    points, _ = await server.qdrant_client.scroll(
        collection_name="media-items",
        limit=limit,
        scroll_filter=flt,
        with_payload=True,
    )
    return points


def _flatten_payload(payload: Mapping[str, JSONValue] | None) -> AggregatedMediaItem:
    """Merge top-level payload fields with the nested data block."""

    data: dict[str, JSONValue] = {}
    if not payload:
        return cast(AggregatedMediaItem, data)
    payload_dict = cast(QdrantMediaPayload, payload)
    base = payload_dict.get("data")
    if isinstance(base, dict):
        data.update(base)
    for key, value in payload_dict.items():
        if key == "data":
            continue
        data[key] = value
    return cast(AggregatedMediaItem, data)


def _normalize_identifier(value: str | int | float | None) -> str | None:
    """Convert mixed identifier formats into a normalized string."""

    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    try:
        normalized_value = str(value)
    except Exception:
        return None
    return normalized_value.strip() or None


def _extract_plex_metadata(media: AggregatedMediaItem) -> PlexMediaMetadata:
    """Return Plex metadata block from an aggregated media item."""

    plex_value = media.get("plex")
    if isinstance(plex_value, dict):
        return cast(PlexMediaMetadata, plex_value)
    return cast(PlexMediaMetadata, {})


async def _get_media_data(identifier: str) -> AggregatedMediaItem:
    """Return the first matching media record's payload."""
    cached = server.cache.get_payload(identifier)
    if cached is not None:
        return cast(AggregatedMediaItem, cached)
    records = await _find_records(identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    payload = _flatten_payload(
        cast(Mapping[str, JSONValue] | None, records[0].payload)
    )
    data = payload

    cache_keys: set[str] = set()

    lookup_key = _normalize_identifier(identifier)
    if lookup_key:
        cache_keys.add(lookup_key)

    plex_data = _extract_plex_metadata(data)
    rating_key = _normalize_identifier(plex_data.get("rating_key"))
    if rating_key:
        cache_keys.add(rating_key)
    guid = _normalize_identifier(plex_data.get("guid"))
    if guid:
        cache_keys.add(guid)

    for source_key in ("imdb", "tmdb", "tvdb"):
        source_value = data.get(source_key)
        if isinstance(source_value, dict):
            source_id = _normalize_identifier(source_value.get("id"))
            if source_id:
                cache_keys.add(source_id)

    for cache_key in cache_keys:
        server.cache.set_payload(cache_key, cast(dict[str, JSONValue], payload))

    if rating_key:
        thumb = plex_data.get("thumb")
        if isinstance(thumb, str) and thumb:
            server.cache.set_poster(rating_key, thumb)
        art = plex_data.get("art")
        if isinstance(art, str) and art:
            server.cache.set_background(rating_key, art)
    return payload


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
            else name
            or product
            or machine_id
            or client_id
            or "Unknown player"
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


def _match_player(query: str, players: Sequence[PlexPlayerMetadata]) -> PlexPlayerMetadata:
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
    def _process_choice(
        choice: str | tuple[str, str, dict[str, Any]]
    ) -> str:
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

    media = await _get_media_data(identifier)
    plex_info = _extract_plex_metadata(media)
    rating_key_value = plex_info.get("rating_key")
    rating_key_normalized = _normalize_identifier(rating_key_value)
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


@server.tool("get-media")
async def get_media(
    identifier: Annotated[
        str,
        Field(
            description="Rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ]
) -> list[AggregatedMediaItem]:
    """Retrieve media items by rating key, IMDb/TMDb ID or title."""
    records = await _find_records(identifier, limit=10)
    return [
        _flatten_payload(cast(Mapping[str, JSONValue] | None, r.payload))
        for r in records
    ]


@server.tool("search-media")
async def search_media(
    query: Annotated[
        str,
        Field(
            description="Search terms for the media library",
            examples=["Matthew McConaughey crime movie"],
        ),
    ],
    limit: Annotated[
        int,
        Field(
            description="Maximum number of results to return",
            ge=1,
            le=50,
            examples=[5],
        ),
    ] = 5,
) -> list[AggregatedMediaItem]:
    """Hybrid similarity search across media items using dense and sparse vectors."""
    dense_doc = models.Document(text=query, model=server.settings.dense_model)
    sparse_doc = models.Document(text=query, model=server.settings.sparse_model)
    reranker = server.reranker
    candidate_limit = limit * 3 if reranker is not None else limit
    prefetch = [
        models.Prefetch(
            query=models.NearestQuery(nearest=dense_doc),
            using="dense",
            limit=candidate_limit,
        ),
        models.Prefetch(
            query=models.NearestQuery(nearest=sparse_doc),
            using="sparse",
            limit=candidate_limit,
        ),
    ]
    res = await server.qdrant_client.query_points(
        collection_name="media-items",
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=prefetch,
        limit=candidate_limit,
        with_payload=True,
    )
    hits = res.points

    async def _prefetch(hit: models.ScoredPoint) -> None:
        data = _flatten_payload(cast(Mapping[str, JSONValue] | None, hit.payload))
        plex_info = _extract_plex_metadata(data)
        rating_key = _normalize_identifier(plex_info.get("rating_key"))
        if rating_key:
            server.cache.set_payload(rating_key, cast(dict[str, JSONValue], data))
            thumb = plex_info.get("thumb")
            if isinstance(thumb, str) and thumb:
                server.cache.set_poster(rating_key, thumb)
            art = plex_info.get("art")
            if isinstance(art, str) and art:
                server.cache.set_background(rating_key, art)

    prefetch_task = asyncio.gather(*[_prefetch(h) for h in hits[:limit]])

    def _rerank(hits: list[models.ScoredPoint]) -> list[models.ScoredPoint]:
        if reranker is None:
            return hits
        docs: list[str] = []
        for h in hits:
            data = _flatten_payload(
                cast(Mapping[str, JSONValue] | None, h.payload)
            )
            plex_info = _extract_plex_metadata(data)
            tmdb_value = data.get("tmdb")
            tmdb_data = tmdb_value if isinstance(tmdb_value, dict) else {}
            parts = [
                data.get("title"),
                data.get("summary"),
                plex_info.get("title"),
                plex_info.get("summary"),
                tmdb_data.get("overview"),
            ]
            directors = data.get("directors") or plex_info.get("directors")
            writers = data.get("writers") or plex_info.get("writers")
            actors = data.get("actors") or plex_info.get("actors")

            def _join_people(values: Any) -> str:
                if isinstance(values, list):
                    names = []
                    for val in values:
                        if isinstance(val, str) and val:
                            names.append(val)
                        elif isinstance(val, dict):
                            tag = val.get("tag") or val.get("name")
                            if tag:
                                names.append(str(tag))
                    return ", ".join(names)
                if isinstance(values, str):
                    return values
                return ""

            director_names = _join_people(directors)
            writer_names = _join_people(writers)
            actor_names = _join_people(actors)
            if director_names:
                parts.append(f"Directed by {director_names}")
            if writer_names:
                parts.append(f"Written by {writer_names}")
            if actor_names:
                parts.append(f"Starring {actor_names}")
            tagline = data.get("tagline") or plex_info.get("tagline")
            if tagline:
                parts.append(tagline if isinstance(tagline, str) else "\n".join(tagline))
            reviews = data.get("reviews")
            if isinstance(reviews, list):
                parts.extend(str(r) for r in reviews if r)
            docs.append(" ".join(p for p in parts if p))
        pairs = [(query, d) for d in docs]
        scores = reranker.predict(pairs)
        for h, s in zip(hits, scores):
            h.score = float(s)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    reranked = await asyncio.to_thread(_rerank, hits)
    await prefetch_task
    return [
        _flatten_payload(cast(Mapping[str, JSONValue] | None, h.payload))
        for h in reranked[:limit]
    ]


@server.tool("query-media")
async def query_media(
    dense_query: Annotated[
        str | None,
        Field(
            description="Text used to generate a dense vector query",
            examples=["british crime comedy"],
        ),
    ] = None,
    sparse_query: Annotated[
        str | None,
        Field(
            description="Text used to generate a sparse vector query",
            examples=["british crime comedy"],
        ),
    ] = None,
    title: Annotated[
        str | None,
        Field(description="Full-text title match", examples=["The Gentlemen"]),
    ] = None,
    type: Annotated[
        str | None,
        Field(
            description="Filter by media type",
            examples=["movie"],
        ),
    ] = None,
    year: Annotated[
        int | None,
        Field(description="Exact release year", examples=[2020]),
    ] = None,
    year_from: Annotated[
        int | None,
        Field(description="Minimum release year", examples=[2018]),
    ] = None,
    year_to: Annotated[
        int | None,
        Field(description="Maximum release year", examples=[2024]),
    ] = None,
    added_after: Annotated[
        int | None,
        Field(
            description="Minimum added_at timestamp (seconds since epoch)",
            examples=[1_700_000_000],
        ),
    ] = None,
    added_before: Annotated[
        int | None,
        Field(
            description="Maximum added_at timestamp (seconds since epoch)",
            examples=[1_760_000_000],
        ),
    ] = None,
    actors: Annotated[
        Sequence[str] | None,
        Field(description="Match actors by name", examples=[["Matthew McConaughey"]]),
    ] = None,
    directors: Annotated[
        Sequence[str] | None,
        Field(description="Match directors by name", examples=[["Guy Ritchie"]]),
    ] = None,
    writers: Annotated[
        Sequence[str] | None,
        Field(description="Match writers by name", examples=[["Guy Ritchie"]]),
    ] = None,
    genres: Annotated[
        Sequence[str] | None,
        Field(description="Match genre tags", examples=[["Action", "Comedy"]]),
    ] = None,
    collections: Annotated[
        Sequence[str] | None,
        Field(description="Match Plex collection names", examples=[["John Wick Collection"]]),
    ] = None,
    show_title: Annotated[
        str | None,
        Field(description="Match the parent show title", examples=["Alien: Earth"]),
    ] = None,
    season_number: Annotated[
        int | None,
        Field(description="Match the season number", examples=[1]),
    ] = None,
    episode_number: Annotated[
        int | None,
        Field(description="Match the episode number", examples=[4]),
    ] = None,
    summary: Annotated[
        str | None,
        Field(description="Full-text search within Plex summaries", examples=["marijuana empire"]),
    ] = None,
    overview: Annotated[
        str | None,
        Field(description="Full-text search within TMDb overviews", examples=["criminal underworld"]),
    ] = None,
    plot: Annotated[
        str | None,
        Field(description="Full-text search within IMDb plots", examples=["drug lord"]),
    ] = None,
    tagline: Annotated[
        str | None,
        Field(description="Full-text search within taglines", examples=["criminal class"]),
    ] = None,
    reviews: Annotated[
        str | None,
        Field(description="Full-text search within review content", examples=["hilarious"]),
    ] = None,
    plex_rating_key: Annotated[
        str | None,
        Field(
            description="Match a specific Plex rating key",
            examples=["49915"],
        ),
    ] = None,
    imdb_id: Annotated[
        str | None,
        Field(description="Match an IMDb identifier", examples=["tt8367814"]),
    ] = None,
    tmdb_id: Annotated[
        int | None,
        Field(description="Match a TMDb identifier", examples=[568467]),
    ] = None,
    limit: Annotated[
        int,
        Field(description="Maximum number of results to return", ge=1, le=50, examples=[5]),
    ] = 5,
) -> list[AggregatedMediaItem]:
    """Run a structured query against indexed payload fields and optional vector searches."""

    def _listify(value: Sequence[str] | str | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return [v for v in value if isinstance(v, str) and v]

    vector_queries: list[tuple[str, models.Document]] = []
    if dense_query:
        vector_queries.append(
            (
                "dense",
                models.Document(text=dense_query, model=server.settings.dense_model),
            )
        )
    if sparse_query:
        vector_queries.append(
            (
                "sparse",
                models.Document(text=sparse_query, model=server.settings.sparse_model),
            )
        )

    must: list[models.FieldCondition] = []
    keyword_prefetch_conditions: list[models.FieldCondition] = []

    if title:
        must.append(models.FieldCondition(key="title", match=models.MatchText(text=title)))
    media_type = type
    if media_type:
        condition = models.FieldCondition(
            key="type", match=models.MatchValue(value=media_type)
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)
    if year is not None:
        must.append(models.FieldCondition(key="year", match=models.MatchValue(value=year)))
    if year_from is not None or year_to is not None:
        rng: dict[str, int] = {}
        if year_from is not None:
            rng["gte"] = year_from
        if year_to is not None:
            rng["lte"] = year_to
        must.append(models.FieldCondition(key="year", range=models.Range(**rng)))
    if added_after is not None or added_before is not None:
        rng_at: dict[str, int] = {}
        if added_after is not None:
            rng_at["gte"] = added_after
        if added_before is not None:
            rng_at["lte"] = added_before
        must.append(models.FieldCondition(key="added_at", range=models.Range(**rng_at)))

    for actor in _listify(actors):
        condition = models.FieldCondition(
            key="actors", match=models.MatchValue(value=actor)
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)
    for director in _listify(directors):
        condition = models.FieldCondition(
            key="directors", match=models.MatchValue(value=director)
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)
    for writer in _listify(writers):
        condition = models.FieldCondition(
            key="writers", match=models.MatchValue(value=writer)
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)
    for genre in _listify(genres):
        condition = models.FieldCondition(
            key="genres", match=models.MatchValue(value=genre)
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)
    for collection in _listify(collections):
        condition = models.FieldCondition(
            key="collections", match=models.MatchValue(value=collection)
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)

    if show_title:
        condition = models.FieldCondition(
            key="show_title", match=models.MatchValue(value=show_title)
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)
    if season_number is not None:
        must.append(
            models.FieldCondition(
                key="season_number", match=models.MatchValue(value=season_number)
            )
        )
    if episode_number is not None:
        must.append(
            models.FieldCondition(
                key="episode_number", match=models.MatchValue(value=episode_number)
            )
        )

    if summary:
        must.append(models.FieldCondition(key="summary", match=models.MatchText(text=summary)))
    if overview:
        must.append(models.FieldCondition(key="overview", match=models.MatchText(text=overview)))
    if plot:
        must.append(models.FieldCondition(key="plot", match=models.MatchText(text=plot)))
    if tagline:
        must.append(models.FieldCondition(key="tagline", match=models.MatchText(text=tagline)))
    if reviews:
        must.append(models.FieldCondition(key="reviews", match=models.MatchText(text=reviews)))

    if plex_rating_key:
        condition = models.FieldCondition(
            key="data.plex.rating_key",
            match=models.MatchValue(value=plex_rating_key),
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)
    if imdb_id:
        condition = models.FieldCondition(
            key="data.imdb.id", match=models.MatchValue(value=imdb_id)
        )
        must.append(condition)
        keyword_prefetch_conditions.append(condition)
    if tmdb_id is not None:
        must.append(
            models.FieldCondition(
                key="data.tmdb.id", match=models.MatchValue(value=tmdb_id)
            )
        )

    filter_obj: models.Filter | None = None
    if must:
        filter_obj = models.Filter(must=must)

    prefetch_filter: models.Filter | None = None
    if keyword_prefetch_conditions:
        prefetch_filter = models.Filter(must=keyword_prefetch_conditions)
        if filter_obj is None:
            filter_obj = models.Filter(must=keyword_prefetch_conditions)

    query_obj: models.Query | None = None
    using_param: str | None = None
    prefetch_param: Sequence[models.Prefetch] | None = None
    if vector_queries:
        candidate_limit = limit * 3 if len(vector_queries) > 1 else limit
        prefetch_entries = [
            models.Prefetch(
                query=models.NearestQuery(nearest=doc),
                using=name,
                limit=candidate_limit,
                filter=prefetch_filter,
            )
            for name, doc in vector_queries
        ]
        if len(prefetch_entries) > 1:
            query_obj = models.FusionQuery(fusion=models.Fusion.RRF)
            using_param = None
            prefetch_param = prefetch_entries
        else:
            prefetch_entry = prefetch_entries[0]
            query_obj = prefetch_entry.query
            using_param = prefetch_entry.using
            prefetch_param = None

    if query_obj is None:
        query_obj = models.SampleQuery(sample=models.Sample.RANDOM)

    res = await server.qdrant_client.query_points(
        collection_name="media-items",
        query=query_obj,
        using=using_param,
        prefetch=prefetch_param,
        query_filter=filter_obj,
        limit=limit,
        with_payload=True,
    )
    return [
        _flatten_payload(cast(Mapping[str, JSONValue] | None, p.payload))
        for p in res.points
    ]


@server.tool("recommend-media")
async def recommend_media(
    identifier: Annotated[
        str,
        Field(
            description="Reference rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ],
    limit: Annotated[
        int,
        Field(
            description="Maximum number of similar items to return",
            ge=1,
            le=50,
            examples=[5],
        ),
    ] = 5,
) -> list[AggregatedMediaItem]:
    """Recommend similar media items based on a reference identifier."""
    record = None
    records = await _find_records(identifier, limit=1)
    if records:
        record = records[0]
    if record is None:
        return []
    rec_query = models.RecommendQuery(
        recommend=models.RecommendInput(positive=[record.id])
    )
    response = await server.qdrant_client.query_points(
        collection_name="media-items",
        query=rec_query,
        limit=limit,
        with_payload=True,
        using="dense",
    )
    return [
        _flatten_payload(cast(Mapping[str, JSONValue] | None, r.payload))
        for r in response.points
    ]


@server.tool("new-movies")
async def new_movies(
    limit: Annotated[
        int,
        Field(
            description="Maximum number of newly added movies to return",
            ge=1,
            le=50,
            examples=[5],
        ),
    ] = 5,
) -> list[AggregatedMediaItem]:
    """Return the most recently added movies."""
    query = models.OrderByQuery(
        order_by=models.OrderBy(key="added_at", direction=models.Direction.DESC)
    )
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key="type", match=models.MatchValue(value="movie")
            )
        ]
    )
    res = await server.qdrant_client.query_points(
        collection_name="media-items",
        query=query,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [
        _flatten_payload(cast(Mapping[str, JSONValue] | None, p.payload))
        for p in res.points
    ]


@server.tool("new-shows")
async def new_shows(
    limit: Annotated[
        int,
        Field(
            description="Maximum number of newly added episodes to return",
            ge=1,
            le=50,
            examples=[5],
        ),
    ] = 5,
) -> list[AggregatedMediaItem]:
    """Return the most recently added TV episodes."""
    query = models.OrderByQuery(
        order_by=models.OrderBy(key="added_at", direction=models.Direction.DESC)
    )
    flt = models.Filter(
        must=[
            models.FieldCondition(
                key="type", match=models.MatchValue(value="episode")
            )
        ]
    )
    res = await server.qdrant_client.query_points(
        collection_name="media-items",
        query=query,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [
        _flatten_payload(cast(Mapping[str, JSONValue] | None, p.payload))
        for p in res.points
    ]


@server.tool("actor-movies")
async def actor_movies(
    actor: Annotated[
        str,
        Field(
            description="Name of the actor to search for",
            examples=["Tom Cruise"],
        ),
    ],
    limit: Annotated[
        int,
        Field(
            description="Maximum number of matching movies to return",
            ge=1,
            le=50,
            examples=[5],
        ),
    ] = 5,
    year_from: Annotated[
        int | None,
        Field(description="Minimum release year", examples=[1990]),
    ] = None,
    year_to: Annotated[
        int | None,
        Field(description="Maximum release year", examples=[1999]),
    ] = None,
) -> list[AggregatedMediaItem]:
    """Return movies featuring the given actor, optionally filtered by release year."""
    must = [
        models.FieldCondition(key="type", match=models.MatchValue(value="movie")),
        models.FieldCondition(key="actors", match=models.MatchValue(value=actor)),
    ]
    if year_from is not None or year_to is not None:
        rng: dict[str, int] = {}
        if year_from is not None:
            rng["gte"] = year_from
        if year_to is not None:
            rng["lte"] = year_to
        must.append(models.FieldCondition(key="year", range=models.Range(**rng)))
    flt = models.Filter(must=must)
    query = models.OrderByQuery(
        order_by=models.OrderBy(key="year", direction=models.Direction.DESC)
    )
    res = await server.qdrant_client.query_points(
        collection_name="media-items",
        query=query,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [
        _flatten_payload(cast(Mapping[str, JSONValue] | None, p.payload))
        for p in res.points
    ]


@server.resource("resource://media-item/{identifier}")
async def media_item(
    identifier: Annotated[
        str,
        Field(
            description="Rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ],
) -> str:
    """Return full metadata for the given media identifier."""
    data = await _get_media_data(identifier)
    return json.dumps(data)


@server.resource("resource://media-ids/{identifier}")
async def media_ids(
    identifier: Annotated[
        str,
        Field(
            description="Rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ],
) -> str:
    """Return external identifiers for the given media item."""
    data = await _get_media_data(identifier)
    plex_info = _extract_plex_metadata(data)
    imdb_value = data.get("imdb")
    imdb_data = imdb_value if isinstance(imdb_value, dict) else {}
    tmdb_value = data.get("tmdb")
    tmdb_data = tmdb_value if isinstance(tmdb_value, dict) else {}
    ids = {
        "rating_key": plex_info.get("rating_key"),
        "imdb": imdb_data.get("id"),
        "tmdb": tmdb_data.get("id"),
        "title": plex_info.get("title"),
    }
    return json.dumps(ids)


@server.resource("resource://media-poster/{identifier}")
async def media_poster(
    identifier: Annotated[
        str,
        Field(
            description="Rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ],
) -> str:
    """Return the poster image URL for the given media identifier."""
    cached = server.cache.get_poster(identifier)
    if cached:
        return cached
    data = await _get_media_data(identifier)
    plex_info = _extract_plex_metadata(data)
    thumb = plex_info.get("thumb")
    if not thumb:
        raise ValueError("Poster not available")
    thumb_str = str(thumb)
    rating_key = _normalize_identifier(plex_info.get("rating_key"))
    if rating_key:
        server.cache.set_poster(rating_key, thumb_str)
    return thumb_str


@server.resource("resource://media-background/{identifier}")
async def media_background(
    identifier: Annotated[
        str,
        Field(
            description="Rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ],
) -> str:
    """Return the background art URL for the given media identifier."""
    cached = server.cache.get_background(identifier)
    if cached:
        return cached
    data = await _get_media_data(identifier)
    plex_info = _extract_plex_metadata(data)
    art = plex_info.get("art")
    if not art:
        raise ValueError("Background not available")
    art_str = str(art)
    rating_key = _normalize_identifier(plex_info.get("rating_key"))
    if rating_key:
        server.cache.set_background(rating_key, art_str)
    return art_str


@server.prompt("media-info")
async def media_info(
    identifier: Annotated[
        str,
        Field(
            description="Rating key, IMDb/TMDb ID, or media title",
            examples=["49915", "tt8367814", "The Gentlemen"],
        ),
    ],
) -> list[Message]:
    """Return a basic description for the given media identifier."""
    data = await _get_media_data(identifier)
    plex_info = _extract_plex_metadata(data)
    title = data.get("title") or plex_info.get("title", "")
    summary = data.get("summary") or plex_info.get("summary", "")
    return [Message(f"{title}: {summary}")]


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
        _r_stub.__name__ = f"resource_{path.replace('/', '_').replace('{', '').replace('}', '')}"
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
    def _register(path: str, method: str, handler: Callable, fn: Callable, name: str) -> None:
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

        async def _rest_resource(request: Request, _uri_template=uri, _resource=resource) -> Response:
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

        handler_name = f"rest_resource_{path.replace('/', '_').replace('{', '').replace('}', '')}"
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
]
