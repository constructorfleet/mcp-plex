"""FastMCP server exposing Plex metadata tools."""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
from collections import OrderedDict
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastmcp.exceptions import NotFoundError
from fastmcp.server import FastMCP
from fastmcp.server.context import Context as FastMCPContext
from pydantic import Field
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# Environment configuration for Qdrant
_QDRANT_URL = os.getenv("QDRANT_URL")
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
_QDRANT_HOST = os.getenv("QDRANT_HOST")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_QDRANT_GRPC_PORT = int(os.getenv("QDRANT_GRPC_PORT", "6334"))
_QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "0") == "1"
_https_env = os.getenv("QDRANT_HTTPS")
_QDRANT_HTTPS = None if _https_env is None else _https_env == "1"

# Embedding model configuration
_DENSE_MODEL_NAME = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
_SPARSE_MODEL_NAME = os.getenv(
    "SPARSE_MODEL", "Qdrant/bm42-all-minilm-l6-v2-attentions"
)

if _QDRANT_URL is None and _QDRANT_HOST is None:
    _QDRANT_URL = ":memory:"

# Instantiate global client
_client = AsyncQdrantClient(
    location=_QDRANT_URL,
    api_key=_QDRANT_API_KEY,
    host=_QDRANT_HOST,
    port=_QDRANT_PORT,
    grpc_port=_QDRANT_GRPC_PORT,
    prefer_grpc=_QDRANT_PREFER_GRPC,
    https=_QDRANT_HTTPS,
)

_USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"
_reranker = None
if _USE_RERANKER and CrossEncoder is not None:
    try:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _reranker = None

server = FastMCP()


_CACHE_SIZE = 128
_payload_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
_poster_cache: OrderedDict[str, str] = OrderedDict()
_background_cache: OrderedDict[str, str] = OrderedDict()


def _cache_set(cache: OrderedDict, key: str, value: Any) -> None:
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    while len(cache) > _CACHE_SIZE:
        cache.popitem(last=False)


def _cache_get(cache: OrderedDict, key: str) -> Any | None:
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    return None


async def _find_records(identifier: str, limit: int = 5) -> list[models.Record]:
    """Locate records matching an identifier or title."""
    # First, try direct ID lookup
    try:
        record_id: Any = int(identifier) if identifier.isdigit() else identifier
        recs = await _client.retrieve("media-items", ids=[record_id], with_payload=True)
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
    points, _ = await _client.scroll(
        collection_name="media-items",
        limit=limit,
        scroll_filter=flt,
        with_payload=True,
    )
    return points


async def _get_media_data(identifier: str) -> dict[str, Any]:
    """Return the first matching media record's payload."""
    cached = _cache_get(_payload_cache, identifier)
    if cached is not None:
        return cached
    records = await _find_records(identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    data = records[0].payload["data"]
    rating_key = str(data.get("plex", {}).get("rating_key"))
    if rating_key:
        _cache_set(_payload_cache, rating_key, data)
        thumb = data.get("plex", {}).get("thumb")
        if thumb:
            _cache_set(_poster_cache, rating_key, thumb)
        art = data.get("plex", {}).get("art")
        if art:
            _cache_set(_background_cache, rating_key, art)
    return data


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
    return [r.payload["data"] for r in records]


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
    dense_doc = models.Document(text=query, model=_DENSE_MODEL_NAME)
    sparse_doc = models.Document(text=query, model=_SPARSE_MODEL_NAME)
    candidate_limit = limit * 3 if _reranker is not None else limit
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
    res = await _client.query_points(
        collection_name="media-items",
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        prefetch=prefetch,
        limit=candidate_limit,
        with_payload=True,
    )
    hits = res.points

    async def _prefetch(hit: models.ScoredPoint) -> None:
        data = hit.payload["data"]
        rating_key = str(data.get("plex", {}).get("rating_key"))
        if rating_key:
            _cache_set(_payload_cache, rating_key, data)
            thumb = data.get("plex", {}).get("thumb")
            if thumb:
                _cache_set(_poster_cache, rating_key, thumb)
            art = data.get("plex", {}).get("art")
            if art:
                _cache_set(_background_cache, rating_key, art)

    prefetch_task = asyncio.gather(*[_prefetch(h) for h in hits[:limit]])

    def _rerank(hits: list[models.ScoredPoint]) -> list[models.ScoredPoint]:
        if _reranker is None:
            return hits
        docs: list[str] = []
        for h in hits:
            data = h.payload["data"]
            parts = [
                data.get("title"),
                data.get("summary"),
                data.get("plex", {}).get("title"),
                data.get("plex", {}).get("summary"),
                data.get("tmdb", {}).get("overview"),
            ]
            docs.append(" ".join(p for p in parts if p))
        pairs = [(query, d) for d in docs]
        scores = _reranker.predict(pairs)
        for h, s in zip(hits, scores):
            h.score = float(s)
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits

    reranked = await asyncio.to_thread(_rerank, hits)
    await prefetch_task
    return [h.payload["data"] for h in reranked[:limit]]


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
    recs = await _client.recommend(
        collection_name="media-items",
        positive=[record.id],
        limit=limit,
        with_payload=True,
        using="dense",
    )
    return [r.payload["data"] for r in recs]


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
    res = await _client.query_points(
        collection_name="media-items",
        query=query,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [p.payload["data"] for p in res.points]


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
    res = await _client.query_points(
        collection_name="media-items",
        query=query,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [p.payload["data"] for p in res.points]


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
    res = await _client.query_points(
        collection_name="media-items",
        query=query,
        query_filter=flt,
        limit=limit,
        with_payload=True,
    )
    return [p.payload["data"] for p in res.points]


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
    cached = _cache_get(_poster_cache, identifier)
    if cached:
        return cached
    data = await _get_media_data(identifier)
    thumb = data.get("plex", {}).get("thumb")
    if not thumb:
        raise ValueError("Poster not available")
    _cache_set(_poster_cache, str(data.get("plex", {}).get("rating_key")), thumb)
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
    cached = _cache_get(_background_cache, identifier)
    if cached:
        return cached
    data = await _get_media_data(identifier)
    art = data.get("plex", {}).get("art")
    if not art:
        raise ValueError("Background not available")
    _cache_set(_background_cache, str(data.get("plex", {}).get("rating_key")), art)
    return art


@server.custom_route("/rest", methods=["GET"])
async def rest_docs(request: Request) -> Response:
    """Serve Swagger UI for REST endpoints."""
    return get_swagger_ui_html(openapi_url="/openapi.json", title="MCP REST API")


def _build_openapi_schema() -> dict[str, Any]:
    app = FastAPI()
    for name, tool in server._tool_manager._tools.items():
        app.post(f"/rest/{name}")(tool.fn)
    return get_openapi(title="MCP REST API", version="1.0.0", routes=app.routes)


_OPENAPI_SCHEMA = _build_openapi_schema()


@server.custom_route("/openapi.json", methods=["GET"])
async def openapi_json(request: Request) -> Response:  # noqa: ARG001
    """Return the OpenAPI schema for REST endpoints."""
    return JSONResponse(_OPENAPI_SCHEMA)


# Dynamically expose tools under `/rest/{tool_name}` while preserving metadata
def _register_rest_tools() -> None:
    for name, tool in server._tool_manager._tools.items():
        async def _rest_tool(request: Request, _tool=tool) -> Response:  # noqa: ARG001
            try:
                arguments = await request.json()
            except Exception:
                arguments = {}
            async with FastMCPContext(fastmcp=server):
                result = await _tool.fn(**arguments)
            return JSONResponse(result)

        _rest_tool.__name__ = f"rest_{name.replace('-', '_')}"
        _rest_tool.__doc__ = tool.fn.__doc__
        _rest_tool.__signature__ = inspect.signature(tool.fn)
        server.custom_route(f"/rest/{name}", methods=["POST"])(_rest_tool)


_register_rest_tools()

@server.custom_route("/rest/prompt/{prompt_name}", methods=["POST"])
async def rest_prompt(request: Request) -> Response:
    """Render a prompt via REST."""
    prompt_name = request.path_params["prompt_name"]
    try:
        arguments = await request.json()
    except Exception:
        arguments = None
    try:
        prompt = await server._prompt_manager.get_prompt(prompt_name)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    async with FastMCPContext(fastmcp=server):
        messages = await prompt.render(arguments)
    return JSONResponse([m.model_dump() for m in messages])


@server.custom_route("/rest/resource/{path:path}", methods=["GET"])
async def rest_resource(request: Request) -> Response:
    """Read a resource via REST."""
    path = request.path_params["path"]
    uri = f"resource://{path}"
    try:
        resource = await server._resource_manager.get_resource(uri)
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    async with FastMCPContext(fastmcp=server):
        data = await server._resource_manager.read_resource(uri)
    if isinstance(data, bytes):
        return Response(content=data, media_type=resource.mime_type)
    try:
        return JSONResponse(json.loads(data), media_type=resource.mime_type)
    except Exception:
        return PlainTextResponse(str(data), media_type=resource.mime_type)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for running the MCP server."""
    global _DENSE_MODEL_NAME, _SPARSE_MODEL_NAME
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
        default=_DENSE_MODEL_NAME,
        help="Dense embedding model name (env: DENSE_MODEL)",
    )
    parser.add_argument(
        "--sparse-model",
        default=_SPARSE_MODEL_NAME,
        help="Sparse embedding model name (env: SPARSE_MODEL)",
    )
    args = parser.parse_args(argv)

    if args.transport != "stdio":
        if not args.bind or not args.port:
            parser.error("--bind and --port are required when transport is not stdio")
    if args.transport == "stdio" and args.mount:
        parser.error("--mount is not allowed when transport is stdio")

    run_kwargs: dict[str, Any] = {}
    if args.transport != "stdio":
        run_kwargs.update({"host": args.bind, "port": args.port})
        if args.mount:
            run_kwargs["path"] = args.mount

    _DENSE_MODEL_NAME = args.dense_model
    _SPARSE_MODEL_NAME = args.sparse_model

    server.run(transport=args.transport, **run_kwargs)


if __name__ == "__main__":
    main()
