"""FastMCP server exposing Plex metadata tools."""
from __future__ import annotations

import os
import json
import asyncio
from typing import Any, Annotated

from fastmcp.server import FastMCP
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client import models
from fastembed import TextEmbedding, SparseTextEmbedding
from pydantic import Field

# Environment configuration for Qdrant
_QDRANT_URL = os.getenv("QDRANT_URL", ":memory:")
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Instantiate global client and embedding models
_client = AsyncQdrantClient(_QDRANT_URL, api_key=_QDRANT_API_KEY)
_dense_model = TextEmbedding("BAAI/bge-small-en-v1.5")
_sparse_model = SparseTextEmbedding("Qdrant/bm42-all-minilm-l6-v2-attentions")

_USE_RERANKER = os.getenv("ENABLE_RERANKER", "0").lower() in {"1", "true", "yes"}
_reranker: Any | None = None


def _get_reranker() -> Any | None:
    """Lazily instantiate the CrossEncoder reranker when enabled."""
    global _reranker
    if not _USE_RERANKER:
        return None
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception:  # pragma: no cover - optional dependency
            return None
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

server = FastMCP()


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
    records = await _find_records(identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    return records[0].payload["data"]


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
    dense_vec = list(_dense_model.embed([query]))[0]
    sparse_vec = _sparse_model.query_embed(query)
    named_dense = models.NamedVector(name="dense", vector=dense_vec)
    sv = models.SparseVector(
        indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
    )
    named_sparse = models.NamedSparseVector(name="sparse", vector=sv)
    reranker = _get_reranker()
    search_limit = limit * 3 if reranker else limit
    hits = await _client.search(
        collection_name="media-items",
        query_vector=named_dense,
        query_sparse_vector=named_sparse,
        limit=search_limit,
        with_payload=True,
    )
    if reranker and hits:
        texts = [h.payload.get("search_text", "") for h in hits]
        pairs = [(query, t) for t in texts]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, lambda: list(reranker.predict(pairs)))
        hits = [h for _, h in sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)]
    return [h.payload["data"] for h in hits[:limit]]


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
    records = await _find_records(identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    thumb = records[0].payload["data"].get("plex", {}).get("thumb")
    if not thumb:
        raise ValueError("Poster not available")
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
    records = await _find_records(identifier, limit=1)
    if not records:
        raise ValueError("Media item not found")
    art = records[0].payload["data"].get("plex", {}).get("art")
    if not art:
        raise ValueError("Background not available")
    return art


if __name__ == "__main__":  # pragma: no cover
    server.run()
