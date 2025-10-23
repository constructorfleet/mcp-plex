"""Media discovery and metadata tools for the Plex MCP server."""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Any, Mapping, Sequence, TYPE_CHECKING, cast

from fastmcp.prompts import Message
from pydantic import Field
from qdrant_client import models

from ...common.types import JSONValue
from .. import media as media_helpers
from ..models import AggregatedMediaItem, MediaSummaryResponse

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .. import PlexServer


def register_media_library_tools(server: "PlexServer") -> None:
    """Register media discovery tools and resources on the provided server."""

    def _media_tool(name: str, *, title: str, operation: str):
        return server.tool(
            name,
            title=title,
            meta={"category": "media-library", "operation": operation},
        )

    @_media_tool("get-media", title="Get media details", operation="lookup")
    async def get_media(
        identifier: Annotated[
            str,
            Field(
                description="Rating key, IMDb/TMDb ID, or media title",
                examples=["49915", "tt8367814", "The Gentlemen"],
            ),
        ],
        summarize_for_llm: Annotated[
            bool,
            Field(
                description=(
                    "Return a compact summary optimized for LLM consumption. "
                    "Set to false to receive the full JSON payload."
                ),
                examples=[True],
            ),
        ] = True,
    ) -> MediaSummaryResponse | list[AggregatedMediaItem]:
        """Retrieve media items by rating key, IMDb/TMDb ID or title."""

        cached_payload = media_helpers._get_cached_payload(server.cache, identifier)

        if cached_payload is not None:
            results = [cached_payload]
        else:
            records = await media_helpers._find_records(server, identifier, limit=10)
            results = [
                media_helpers._flatten_payload(
                    cast(Mapping[str, JSONValue] | None, r.payload)
                )
                for r in records
            ]
        if summarize_for_llm:
            return media_helpers.summarize_media_items_for_llm(results)
        return results

    @_media_tool("search-media", title="Search media library", operation="search")
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
        summarize_for_llm: Annotated[
            bool,
            Field(
                description=(
                    "Return a compact summary optimized for LLM consumption. "
                    "Set to false to receive the full JSON payload."
                ),
                examples=[True],
            ),
        ] = True,
    ) -> MediaSummaryResponse | list[AggregatedMediaItem]:
        """Hybrid similarity search across media items using dense and sparse vectors."""

        dense_doc = models.Document(text=query, model=server.settings.dense_model)
        sparse_doc = models.Document(text=query, model=server.settings.sparse_model)
        reranker = await server.ensure_reranker()
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
            data = media_helpers._flatten_payload(
                cast(Mapping[str, JSONValue] | None, hit.payload)
            )
            plex_info = media_helpers._extract_plex_metadata(data)
            cache_keys = media_helpers._collect_cache_keys(data, plex_info)
            thumb = plex_info.get("thumb")
            art = plex_info.get("art")
            for cache_key in cache_keys:
                server.cache.set_payload(cache_key, cast(dict[str, JSONValue], data))
                if isinstance(thumb, str) and thumb:
                    server.cache.set_poster(cache_key, thumb)
                if isinstance(art, str) and art:
                    server.cache.set_background(cache_key, art)

        prefetch_task = asyncio.gather(*[_prefetch(h) for h in hits[:limit]])

        def _rerank(hits: list[models.ScoredPoint]) -> list[models.ScoredPoint]:
            if reranker is None:
                return hits
            docs: list[str] = []
            for h in hits:
                data = media_helpers._flatten_payload(
                    cast(Mapping[str, JSONValue] | None, h.payload)
                )
                plex_info = media_helpers._extract_plex_metadata(data)
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
                    parts.append(
                        tagline if isinstance(tagline, str) else "\n".join(tagline)
                    )
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
        results = [
            media_helpers._flatten_payload(
                cast(Mapping[str, JSONValue] | None, h.payload)
            )
            for h in reranked[:limit]
        ]
        if summarize_for_llm:
            return media_helpers.summarize_media_items_for_llm(results)
        return results

    @_media_tool("query-media", title="Query media library", operation="query")
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
            Field(
                default=None,
                description="Minimum release year",
                examples=[2018],
                json_schema_extra={"nullable": True, "type": "integer"},
            ),
        ] = None,
        year_to: Annotated[
            int | None,
            Field(
                default=None,
                description="Maximum release year",
                examples=[2024],
                json_schema_extra={"nullable": True, "type": "integer"},
            ),
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
            Field(
                description="Match actors by name", examples=[["Matthew McConaughey"]]
            ),
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
            Field(
                description="Match Plex collection names",
                examples=[["John Wick Collection"]],
            ),
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
            Field(
                description="Full-text search within Plex summaries",
                examples=["marijuana empire"],
            ),
        ] = None,
        overview: Annotated[
            str | None,
            Field(
                description="Full-text search within TMDb overviews",
                examples=["criminal underworld"],
            ),
        ] = None,
        plot: Annotated[
            str | None,
            Field(
                description="Full-text search within IMDb plots",
                examples=["drug lord"],
            ),
        ] = None,
        tagline: Annotated[
            str | None,
            Field(
                description="Full-text search within taglines",
                examples=["criminal class"],
            ),
        ] = None,
        reviews: Annotated[
            str | None,
            Field(
                description="Full-text search within review content",
                examples=["hilarious"],
            ),
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
        similar_to: Annotated[
            str | int | Sequence[str | int] | None,
            Field(
                description=(
                    "Recommend candidates similar to Plex rating keys, "
                    "IMDb/TMDb identifiers, or titles"
                ),
                examples=[[49915], "tt8367814", 568467, "The Gentlemen"],
            ),
        ] = None,
        limit: Annotated[
            int,
            Field(
                description="Maximum number of results to return",
                ge=1,
                le=50,
                examples=[5],
            ),
        ] = 5,
        summarize_for_llm: Annotated[
            bool,
            Field(
                description=(
                    "Return a compact summary optimized for LLM consumption. "
                    "Set to false to receive the full JSON payload."
                ),
                examples=[True],
            ),
        ] = True,
    ) -> MediaSummaryResponse | list[AggregatedMediaItem]:
        """Run a structured query against indexed payload fields and optional vector searches."""

        def _listify(
            value: Sequence[str | int] | str | int | None,
        ) -> list[str]:
            if value is None:
                return []
            if isinstance(value, (str, int)):
                text = str(value).strip()
                return [text] if text else []
            items: list[str] = []
            for entry in value:
                if isinstance(entry, (str, int)):
                    text = str(entry).strip()
                    if text:
                        items.append(text)
            return items

        def _finalize(
            items: list[AggregatedMediaItem],
        ) -> MediaSummaryResponse | list[AggregatedMediaItem]:
            if summarize_for_llm:
                return media_helpers.summarize_media_items_for_llm(items)
            return items

        def _append_range_condition(
            conditions: list[models.FieldCondition],
            key: str,
            *,
            gte: int | None,
            lte: int | None,
        ) -> None:
            if gte is None and lte is None:
                return
            range_kwargs: dict[str, int] = {}
            if gte is not None:
                range_kwargs["gte"] = gte
            if lte is not None:
                range_kwargs["lte"] = lte
            conditions.append(
                models.FieldCondition(key=key, range=models.Range(**range_kwargs))
            )

        vector_queries: list[tuple[str, models.Document]] = []
        positive_point_ids: list[Any] = []
        similar_identifiers = _listify(similar_to)
        if similar_identifiers:
            for identifier in similar_identifiers:
                records = await media_helpers._find_records(
                    server, identifier, limit=1
                )
                for record in records:
                    if record.id is not None:
                        positive_point_ids.append(record.id)
            if not positive_point_ids:
                return _finalize([])
        if not positive_point_ids:
            if dense_query:
                vector_queries.append(
                    (
                        "dense",
                        models.Document(
                            text=dense_query, model=server.settings.dense_model
                        ),
                    )
                )
            if sparse_query:
                vector_queries.append(
                    (
                        "sparse",
                        models.Document(
                            text=sparse_query, model=server.settings.sparse_model
                        ),
                    )
                )

        must: list[models.FieldCondition] = []
        keyword_prefetch_conditions: list[models.FieldCondition] = []

        if title:
            must.append(
                models.FieldCondition(key="title", match=models.MatchText(text=title))
            )
        media_type = type
        if media_type:
            condition = models.FieldCondition(
                key="type", match=models.MatchValue(value=media_type)
            )
            must.append(condition)
            keyword_prefetch_conditions.append(condition)
        if year is not None:
            must.append(
                models.FieldCondition(key="year", match=models.MatchValue(value=year))
            )
        _append_range_condition(must, "year", gte=year_from, lte=year_to)
        _append_range_condition(
            must, "added_at", gte=added_after, lte=added_before
        )

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
            must.append(
                models.FieldCondition(
                    key="summary", match=models.MatchText(text=summary)
                )
            )
        if overview:
            must.append(
                models.FieldCondition(
                    key="overview", match=models.MatchText(text=overview)
                )
            )
        if plot:
            must.append(
                models.FieldCondition(key="plot", match=models.MatchText(text=plot))
            )
        if tagline:
            must.append(
                models.FieldCondition(
                    key="tagline", match=models.MatchText(text=tagline)
                )
            )
        if reviews:
            must.append(
                models.FieldCondition(
                    key="reviews", match=models.MatchText(text=reviews)
                )
            )

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
        prefetch_entries: list[models.Prefetch] = []
        if positive_point_ids:
            recommend_query = models.RecommendQuery(
                recommend=models.RecommendInput(positive=positive_point_ids)
            )
            prefetch_entries.append(
                models.Prefetch(
                    query=recommend_query,
                    using="dense",
                    limit=limit,
                    filter=prefetch_filter,
                )
            )
        if not positive_point_ids and vector_queries:
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

        if prefetch_entries:
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
        results = [
            media_helpers._flatten_payload(
                cast(Mapping[str, JSONValue] | None, p.payload)
            )
            for p in res.points
        ]
        return _finalize(results)

    @_media_tool(
        "recommend-media",
        title="Recommend similar media",
        operation="recommend",
    )
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
        summarize_for_llm: Annotated[
            bool,
            Field(
                description=(
                    "Return a compact summary optimized for LLM consumption. "
                    "Set to false to receive the full JSON payload."
                ),
                examples=[True],
            ),
        ] = True,
    ) -> MediaSummaryResponse | list[AggregatedMediaItem]:
        """Recommend similar media items based on a reference identifier."""

        record = None
        records = await media_helpers._find_records(server, identifier, limit=1)
        if records:
            record = records[0]
        if record is None:
            if summarize_for_llm:
                return media_helpers.summarize_media_items_for_llm([])
            return []
        def _normalize_rating_key(value: object) -> str | None:
            try:
                text = str(value).strip()
            except Exception:  # noqa: BLE001 - guard against unusual reprs
                return None
            return text or None

        positive_rating_key: str | None = None
        if isinstance(record.payload, Mapping):
            original_data = media_helpers._flatten_payload(
                cast(Mapping[str, JSONValue] | None, record.payload)
            )
            plex_info = media_helpers._extract_plex_metadata(original_data)
            key_value = plex_info.get("rating_key")
            positive_rating_key = _normalize_rating_key(key_value)

        rec_query = models.RecommendQuery(
            recommend=models.RecommendInput(positive=[record.id])
        )
        raw_watched_keys = await server.get_watched_rating_keys()
        watched_keys = {
            normalized
            for key in raw_watched_keys
            if (normalized := _normalize_rating_key(key)) is not None
        }
        exclusion_values: set[str] = set(watched_keys)
        query_filter = None
        if exclusion_values:
            query_filter = models.Filter(
                must_not=[
                    models.FieldCondition(
                        key="data.plex.rating_key",
                        match=models.MatchAny(any=list(exclusion_values)),
                    )
                ]
            )

        extra_candidates = 1 + min(len(exclusion_values), 20) if exclusion_values else 1
        query_limit = min(limit + extra_candidates, 100)

        response = await server.qdrant_client.query_points(
            collection_name="media-items",
            query=rec_query,
            limit=query_limit,
            with_payload=True,
            using="dense",
            query_filter=query_filter,
        )
        results: list[AggregatedMediaItem] = []
        for r in response.points:
            data = media_helpers._flatten_payload(
                cast(Mapping[str, JSONValue] | None, r.payload)
            )
            if not watched_keys:
                results.append(data)
                continue
            plex_info = media_helpers._extract_plex_metadata(data)
            rating_key = plex_info.get("rating_key")
            normalized = _normalize_rating_key(rating_key)
            if normalized and normalized in watched_keys:
                continue
            if normalized and positive_rating_key and normalized == positive_rating_key:
                continue
            results.append(data)
        if len(results) > limit:
            results = results[:limit]
        if summarize_for_llm:
            return media_helpers.summarize_media_items_for_llm(results)
        return results

    @_media_tool("new-movies", title="Newest movies", operation="recent-movies")
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
        summarize_for_llm: Annotated[
            bool,
            Field(
                description=(
                    "Return a compact summary optimized for LLM consumption. "
                    "Set to false to receive the full JSON payload."
                ),
                examples=[True],
            ),
        ] = True,
    ) -> MediaSummaryResponse | list[AggregatedMediaItem]:
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
        results = [
            media_helpers._flatten_payload(
                cast(Mapping[str, JSONValue] | None, p.payload)
            )
            for p in res.points
        ]
        if summarize_for_llm:
            return media_helpers.summarize_media_items_for_llm(results)
        return results

    @_media_tool("new-shows", title="Newest episodes", operation="recent-episodes")
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
        summarize_for_llm: Annotated[
            bool,
            Field(
                description=(
                    "Return a compact summary optimized for LLM consumption. "
                    "Set to false to receive the full JSON payload."
                ),
                examples=[True],
            ),
        ] = True,
    ) -> MediaSummaryResponse | list[AggregatedMediaItem]:
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
        results = [
            media_helpers._flatten_payload(
                cast(Mapping[str, JSONValue] | None, p.payload)
            )
            for p in res.points
        ]
        if summarize_for_llm:
            return media_helpers.summarize_media_items_for_llm(results)
        return results

    @_media_tool(
        "actor-movies",
        title="Movies by actor",
        operation="actor-filmography",
    )
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
        summarize_for_llm: Annotated[
            bool,
            Field(
                description=(
                    "Return a compact summary optimized for LLM consumption. "
                    "Set to false to receive the full JSON payload."
                ),
                examples=[True],
            ),
        ] = True,
    ) -> MediaSummaryResponse | list[AggregatedMediaItem]:
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
        results = [
            media_helpers._flatten_payload(
                cast(Mapping[str, JSONValue] | None, p.payload)
            )
            for p in res.points
        ]
        if summarize_for_llm:
            return media_helpers.summarize_media_items_for_llm(results)
        return results

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

        data = await media_helpers._get_media_data(server, identifier)
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

        data = await media_helpers._get_media_data(server, identifier)
        plex_info = media_helpers._extract_plex_metadata(data)
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
        data = await media_helpers._get_media_data(server, identifier)
        plex_info = media_helpers._extract_plex_metadata(data)
        thumb = plex_info.get("thumb")
        if not thumb:
            raise ValueError("Poster not available")
        thumb_str = str(thumb)
        cache_keys = media_helpers._collect_cache_keys(data, plex_info, identifier)
        cache_keys = media_helpers._ensure_rating_key_cached(cache_keys, plex_info)
        media_helpers._cache_media_artwork(server.cache, cache_keys, plex_info)
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
        data = await media_helpers._get_media_data(server, identifier)
        plex_info = media_helpers._extract_plex_metadata(data)
        art = plex_info.get("art")
        if not art:
            raise ValueError("Background not available")
        art_str = str(art)
        cache_keys = media_helpers._collect_cache_keys(data, plex_info, identifier)
        cache_keys = media_helpers._ensure_rating_key_cached(cache_keys, plex_info)
        media_helpers._cache_media_artwork(server.cache, cache_keys, plex_info)
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

        data = await media_helpers._get_media_data(server, identifier)
        plex_info = media_helpers._extract_plex_metadata(data)
        title = data.get("title") or plex_info.get("title", "")
        summary = data.get("summary") or plex_info.get("summary", "")
        return [Message(f"{title}: {summary}")]
