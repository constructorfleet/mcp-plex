"""FastMCP server exposing Plex metadata tools."""
from __future__ import annotations

import asyncio
import json
import os
from collections import OrderedDict
from typing import Annotated, Any

from fastembed import SparseTextEmbedding, TextEmbedding
from fastmcp.server import FastMCP
from pydantic import Field
from qdrant_client import models
from qdrant_client.async_qdrant_client import AsyncQdrantClient

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

# Environment configuration for Qdrant
_QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Instantiate global client and embedding models
_client = AsyncQdrantClient(_QDRANT_URL, api_key=_QDRANT_API_KEY)
_dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
_sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

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
        filter=flt,
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
    dense_task = asyncio.to_thread(lambda: list(_dense_model.embed([query]))[0])
    sparse_task = asyncio.to_thread(lambda: next(_sparse_model.query_embed(query)))
    dense_vec, sparse_vec = await asyncio.gather(dense_task, sparse_task)
    named_dense = models.NamedVector(name="dense", vector=dense_vec)
    sv = models.SparseVector(
        indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
    )
    named_sparse = models.NamedSparseVector(name="sparse", vector=sv)
    candidate_limit = limit * 3 if _reranker is not None else limit
    hits = await _client.search(
        collection_name="media-items",
        query_vector=named_dense,
        query_sparse_vector=named_sparse,
        limit=candidate_limit,
        with_payload=True,
    )

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
    )
    return [r.payload["data"] for r in recs]


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


if __name__ == "__main__":
    server.run()
