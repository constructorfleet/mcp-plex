"""FastMCP server exposing Plex metadata tools."""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
from typing import Annotated, Any, Callable

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastmcp.prompts import Message
from fastmcp.server import FastMCP
from fastmcp.server.context import Context as FastMCPContext
from pydantic import Field
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from .cache import MediaCache
from .config import Settings

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None


settings = Settings()


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

        async def _lifespan(app: FastMCP):  # noqa: ARG001
            yield
            await self.close()

        super().__init__(lifespan=_lifespan)
        self._reranker: CrossEncoder | None = None
        self._reranker_loaded = False
        self.cache = MediaCache(self.settings.cache_size)

    async def close(self) -> None:
        await self.qdrant_client.close()

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
            except Exception:
                self._reranker = None
            self._reranker_loaded = True
        return self._reranker


server = PlexServer(settings=settings)


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


async def _get_media_data(identifier: str) -> dict[str, Any]:
    """Return the first matching media record's payload."""
    cached = server.cache.get_payload(identifier)
    if cached is not None:
        return cached
    records = await _find_records(identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    data = records[0].payload["data"]
    rating_key = str(data.get("plex", {}).get("rating_key"))
    if rating_key:
        server.cache.set_payload(rating_key, data)
        thumb = data.get("plex", {}).get("thumb")
        if thumb:
            server.cache.set_poster(rating_key, thumb)
        art = data.get("plex", {}).get("art")
        if art:
            server.cache.set_background(rating_key, art)
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
        data = hit.payload["data"]
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
        scores = reranker.predict(pairs)
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
    recs = await server.qdrant_client.recommend(
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
    res = await server.qdrant_client.query_points(
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
    res = await server.qdrant_client.query_points(
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
    res = await server.qdrant_client.query_points(
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


def _build_openapi_schema() -> dict[str, Any]:
    app = FastAPI()
    for name, tool in server._tool_manager._tools.items():
        app.post(f"/rest/{name}")(tool.fn)
    for name, prompt in server._prompt_manager._prompts.items():
        async def _p_stub(**kwargs):  # noqa: ARG001
            pass
        _p_stub.__name__ = f"prompt_{name.replace('-', '_')}"
        _p_stub.__doc__ = prompt.fn.__doc__
        _p_stub.__signature__ = inspect.signature(prompt.fn).replace(
            return_annotation=Any
        )
        app.post(f"/rest/prompt/{name}")(_p_stub)
    for uri, resource in server._resource_manager._templates.items():
        path = uri.replace("resource://", "")
        async def _r_stub(**kwargs):  # noqa: ARG001
            pass
        _r_stub.__name__ = f"resource_{path.replace('/', '_').replace('{', '').replace('}', '')}"
        _r_stub.__doc__ = resource.fn.__doc__
        _r_stub.__signature__ = inspect.signature(resource.fn).replace(
            return_annotation=Any
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
        handler.__signature__ = inspect.signature(fn).replace(return_annotation=Any)
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

    server.settings.dense_model = args.dense_model
    server.settings.sparse_model = args.sparse_model

    server.run(transport=args.transport, **run_kwargs)


if __name__ == "__main__":
    main()
