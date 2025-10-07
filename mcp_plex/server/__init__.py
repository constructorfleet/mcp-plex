"""FastMCP server exposing Plex metadata tools."""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import inspect
import json
import logging
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, ForwardRef, TypedDict, cast

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastmcp.prompts import Message
from fastmcp.server import FastMCP
from fastmcp.server.context import Context as FastMCPContext
from plexapi.exceptions import PlexApiException
from plexapi.server import PlexServer as PlexServerClient
from pydantic import BaseModel, Field, create_model
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from rapidfuzz import fuzz, process

from ..common.cache import MediaCache
from ..common.types import JSONValue
from .config import Settings


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
        annotation = (
            parameter.annotation
            if parameter.annotation is not inspect._empty
            else object
        )
        default = (
            parameter.default
            if parameter.default is not inspect._empty
            else ...
        )
        fields[param_name] = (annotation, default)

    if not fields:
        return None

    model_name = "".join(part.capitalize() for part in name.replace("-", "_").split("_"))
    model_name = f"{model_name or 'Request'}Request"
    request_model = create_model(model_name, **fields)  # type: ignore[arg-type]
    return request_model


def _sanitize_return_annotation(annotation: object) -> object:
    """Return a return annotation safe for FastAPI schema generation."""

    if annotation is inspect.Signature.empty:
        return inspect.Signature.empty
    if isinstance(annotation, str):
        return inspect.Signature.empty
    if isinstance(annotation, ForwardRef):
        return inspect.Signature.empty
    return annotation


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


def _flatten_payload(payload: Mapping[str, JSONValue] | None) -> dict[str, JSONValue]:
    """Merge top-level payload fields with the nested data block."""

    data: dict[str, JSONValue] = {}
    if not payload:
        return data
    base = payload.get("data")
    if isinstance(base, dict):
        data.update(base)
    for key, value in payload.items():
        if key == "data":
            continue
        data[key] = value
    return data


async def _get_media_data(identifier: str) -> dict[str, JSONValue]:
    """Return the first matching media record's payload."""
    cached = server.cache.get_payload(identifier)
    if cached is not None:
        return cached
    records = await _find_records(identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    payload = _flatten_payload(
        cast(Mapping[str, JSONValue] | None, records[0].payload)
    )
    data = payload

    def _normalize_identifier(value: str | int | float | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip()
            return normalized or None
        try:
            return str(value)
        except Exception:
            return None

    cache_keys: set[str] = set()

    lookup_key = _normalize_identifier(identifier)
    if lookup_key:
        cache_keys.add(lookup_key)

    plex_value = data.get("plex")
    plex_data: dict[str, JSONValue] = (
        plex_value if isinstance(plex_value, dict) else {}
    )
    rating_value = plex_data.get("rating_key")
    rating_key = _normalize_identifier(
        rating_value if isinstance(rating_value, (str, int, float)) else None
    )
    if rating_key:
        cache_keys.add(rating_key)
    guid_value = plex_data.get("guid")
    guid = _normalize_identifier(guid_value if isinstance(guid_value, str) else None)
    if guid:
        cache_keys.add(guid)

    for source_key in ("imdb", "tmdb", "tvdb"):
        source_value = data.get(source_key)
        if isinstance(source_value, dict):
            raw_source_id = source_value.get("id")
            source_id = _normalize_identifier(
                raw_source_id
                if isinstance(raw_source_id, (str, int, float))
                else None
            )
            if source_id:
                cache_keys.add(source_id)

    for cache_key in cache_keys:
        server.cache.set_payload(cache_key, payload)

    if rating_key:
        thumb = plex_data.get("thumb")
        if thumb:
            server.cache.set_poster(rating_key, thumb)
        art = plex_data.get("art")
        if art:
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


async def _get_plex_players() -> list[dict[str, Any]]:
    """Return Plex players available for playback commands."""

    plex_client = await _get_plex_client()

    def _load_clients() -> list[Any]:
        return list(plex_client.clients())

    raw_clients = await asyncio.to_thread(_load_clients)
    aliases = server.settings.plex_player_aliases
    players: list[dict[str, Any]] = []

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
        port = getattr(client, "port", None)
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

        players.append(
            {
                "name": name,
                "product": product,
                "display_name": display_name,
                "friendly_names": friendly_names,
                "machine_identifier": machine_id,
                "client_identifier": client_id,
                "address": address,
                "port": port,
                "provides": provides,
                "client": client,
            }
        )

    return players


_FUZZY_MATCH_THRESHOLD = 70


def _match_player(query: str, players: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Locate a Plex player by friendly name or identifier."""

    normalized_query = query.strip()
    normalized = normalized_query.lower()
    if not normalized_query:
        raise ValueError(f"Player '{query}' not found")

    candidate_entries: list[tuple[str, str, dict[str, Any]]] = []
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
    rating_key: str, player: dict[str, Any], offset_seconds: int
) -> None:
    """Send a playback command to the selected player."""

    if "player" not in player.get("provides", set()):
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
        plex_client.playMedia(
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
    plex_info = media.get("plex") or {}
    rating_key = plex_info.get("rating_key")
    if not rating_key:
        raise ValueError("Media item is missing a Plex rating key")

    players = await _get_plex_players()
    target = _match_player(player, players)
    await _start_playback(str(rating_key), target, offset_seconds or 0)

    return {
        "player": target.get("display_name"),
        "rating_key": str(rating_key),
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
) -> list[dict[str, Any]]:
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
) -> list[dict[str, Any]]:
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
        rating_key = str(data.get("plex", {}).get("rating_key"))
        if rating_key:
            server.cache.set_payload(rating_key, data)
            thumb = data.get("plex", {}).get("thumb")
            if thumb:
                server.cache.set_poster(rating_key, thumb)
            art = data.get("plex", {}).get("art")
            if art:
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
            parts = [
                data.get("title"),
                data.get("summary"),
                data.get("plex", {}).get("title"),
                data.get("plex", {}).get("summary"),
                data.get("tmdb", {}).get("overview"),
            ]
            directors = data.get("directors") or data.get("plex", {}).get("directors")
            writers = data.get("writers") or data.get("plex", {}).get("writers")
            actors = data.get("actors") or data.get("plex", {}).get("actors")

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
            tagline = data.get("tagline") or data.get("plex", {}).get("tagline")
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
) -> list[dict[str, Any]]:
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
) -> list[dict[str, Any]]:
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
) -> list[dict[str, Any]]:
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
) -> list[dict[str, Any]]:
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
) -> list[dict[str, Any]]:
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
    ids = {
        "rating_key": data.get("plex", {}).get("rating_key"),
        "imdb": data.get("imdb", {}).get("id"),
        "tmdb": data.get("tmdb", {}).get("id"),
        "title": data.get("plex", {}).get("title"),
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
    thumb = data.get("plex", {}).get("thumb")
    if not thumb:
        raise ValueError("Poster not available")
    server.cache.set_poster(
        str(data.get("plex", {}).get("rating_key")), thumb
    )
    return thumb


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
    art = data.get("plex", {}).get("art")
    if not art:
        raise ValueError("Background not available")
    server.cache.set_background(
        str(data.get("plex", {}).get("rating_key")), art
    )
    return art


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
    title = data.get("title") or data.get("plex", {}).get("title", "")
    summary = data.get("summary") or data.get("plex", {}).get("summary", "")
    return [Message(f"{title}: {summary}")]


@server.custom_route("/rest", methods=["GET"])
async def rest_docs(request: Request) -> Response:
    """Serve Swagger UI for REST endpoints."""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MCP REST API")


def _build_openapi_schema() -> dict[str, object]:
    app = FastAPI()
    for name, tool in server._tool_manager._tools.items():
        request_model = _request_model(name, tool.fn)
        tool_signature = inspect.signature(tool.fn)

        if request_model is None:
            app.post(f"/rest/{name}")(tool.fn)
            continue

        async def _tool_stub(payload: request_model) -> None:  # type: ignore[name-defined]
            pass

        _tool_stub.__name__ = f"tool_{name.replace('-', '_')}"
        _tool_stub.__doc__ = tool.fn.__doc__
        tool_return = _sanitize_return_annotation(tool_signature.return_annotation)
        _tool_stub.__signature__ = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    "payload",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=request_model,
                )
            ],
            return_annotation=tool_return,
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
            _p_stub.__signature__ = prompt_signature
        else:
            prompt_return = _sanitize_return_annotation(
                prompt_signature.return_annotation
            )
            _p_stub.__signature__ = inspect.Signature(
                parameters=[
                    inspect.Parameter(
                        "payload",
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        annotation=request_model,
                    )
                ],
                return_annotation=prompt_return,
            )
        app.post(f"/rest/prompt/{name}")(_p_stub)
    for uri, resource in server._resource_manager._templates.items():
        path = uri.replace("resource://", "")
        async def _r_stub(**kwargs):  # noqa: ARG001
            pass
        _r_stub.__name__ = f"resource_{path.replace('/', '_').replace('{', '').replace('}', '')}"
        _r_stub.__doc__ = resource.fn.__doc__
        _r_stub.__signature__ = inspect.signature(resource.fn)
        app.get(f"/rest/resource/{path}")(_r_stub)
    return get_openapi(title="MCP REST API", version="1.0.0", routes=app.routes)


_OPENAPI_SCHEMA = _build_openapi_schema()


@server.custom_route("/openapi.json", methods=["GET"])
async def openapi_json(request: Request) -> Response:  # noqa: ARG001
    """Return the OpenAPI schema for REST endpoints."""
    return JSONResponse(_OPENAPI_SCHEMA)



def _register_rest_endpoints() -> None:
    def _register(
        path: str,
        method: str,
        handler: Callable[..., object],
        fn: Callable[..., object],
        name: str,
    ) -> None:
        handler.__name__ = name
        handler.__doc__ = fn.__doc__
        original_signature = inspect.signature(fn)
        handler.__signature__ = original_signature.replace(
            return_annotation=_sanitize_return_annotation(
                original_signature.return_annotation
            )
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


class RunConfig(TypedDict, total=False):
    host: str
    port: int
    path: str


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running the MCP server."""
    parser = argparse.ArgumentParser(description="Run the MCP server")
    parser.add_argument("--bind", help="Host address to bind to")
    parser.add_argument("--port", type=int, help="Port to listen on")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport protocol to use",
    )
    parser.add_argument("--mount", help="Mount path for HTTP transports")
    parser.add_argument(
        "--dense-model",
        default=server.settings.dense_model,
        help="Dense embedding model name (env: DENSE_MODEL)",
    )
    parser.add_argument(
        "--sparse-model",
        default=server.settings.sparse_model,
        help="Sparse embedding model name (env: SPARSE_MODEL)",
    )
    args = parser.parse_args(argv)

    env_transport = os.getenv("MCP_TRANSPORT")
    env_host = os.getenv("MCP_HOST") if os.getenv("MCP_HOST") is not None else os.getenv("MCP_BIND")
    env_port = os.getenv("MCP_PORT")
    env_mount = os.getenv("MCP_MOUNT")

    transport = env_transport or args.transport
    valid_transports = {"stdio", "sse", "streamable-http"}
    if transport not in valid_transports:
        parser.error(
            "transport must be one of stdio, sse, or streamable-http (via --transport or MCP_TRANSPORT)"
        )

    host = env_host or args.bind
    port: int | None
    if env_port is not None:
        try:
            port = int(env_port)
        except ValueError:
            parser.error("MCP_PORT must be an integer")
    else:
        port = args.port

    mount = env_mount or args.mount

    if transport != "stdio":
        if host is None or port is None:
            parser.error(
                "--bind/--port or MCP_HOST/MCP_PORT are required when transport is not stdio"
            )
    if transport == "stdio" and mount:
        parser.error("--mount or MCP_MOUNT is not allowed when transport is stdio")

    run_config: RunConfig = {}
    if transport != "stdio":
        assert host is not None
        assert port is not None
        run_config.update({"host": host, "port": port})
        if mount:
            run_config["path"] = mount

    server.settings.dense_model = args.dense_model
    server.settings.sparse_model = args.sparse_model

    server.run(transport=transport, **run_config)


if __name__ == "__main__":
    main()
