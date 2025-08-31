"""FastMCP server exposing Plex metadata tools."""
from __future__ import annotations

import os
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

server = FastMCP()


async def _find_records(identifier: str, limit: int = 5) -> list[models.Record]:
    """Locate records matching an identifier or title."""
    # First, try direct ID lookup
    try:
        record_id: Any = int(identifier) if identifier.isdigit() else identifier
        recs = await _client.retrieve("media-items", ids=[record_id], with_payload=True)
        if recs:
            return recs
    except Exception:  # pragma: no cover - Qdrant retrieval failures fall back to search
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
    hits = await _client.search(
        collection_name="media-items",
        query_vector=named_dense,
        query_sparse_vector=named_sparse,
        limit=limit,
        with_payload=True,
    )
    return [h.payload["data"] for h in hits]


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


if __name__ == "__main__":  # pragma: no cover
    server.run()
