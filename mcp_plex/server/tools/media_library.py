"""Media discovery and metadata tools for the Plex MCP server."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Annotated, Any, Mapping, Sequence, TYPE_CHECKING, cast

from fastmcp.prompts import Message
from pydantic import Field
from qdrant_client import models
from rapidfuzz import fuzz

from ...common.types import JSONValue
from .. import media as media_helpers
from ..models import (
    AggregatedMediaItem,
    AggregatedMediaItemModel,
    MediaSummaryResponseModel,
)

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from .. import PlexServer


logger = logging.getLogger(__name__)


def _strip_leading_article(title: str | None) -> str | None:
    """Remove leading articles (The, A, An, etc.) from a title for search purposes."""
    if not title:
        return title
    # Regex for common English articles
    return re.sub(r"^(the|a|an)\s+", "", title, flags=re.IGNORECASE).strip() or title


def _listify(
        value: Sequence[str | int] | str | int | None,
    ) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (str, int)):
            text = str(value).strip()
            return [text] if text else []
        items_list: list[str] = []
        for entry in value:
            if isinstance(entry, (str, int)):
                text = str(entry).strip()
                if text:
                    items_list.append(text)
        return items_list


def _build_rerank_document(media: AggregatedMediaItem) -> str:
        plex_info = media_helpers._extract_plex_metadata(media)
        segments: list[str] = []

        def _append_text(*values: Any) -> None:
            for value in values:
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        segments.append(text)
                elif isinstance(value, Sequence) and not isinstance(
                    value, (str, bytes, bytearray)
                ):
                    for entry in value:
                        if isinstance(entry, str):
                            text = entry.strip()
                            if text:
                                segments.append(text)

        _append_text(
            media.get("title"),
            plex_info.get("title"),
            media.get("show_title"),
            plex_info.get("grandparent_title"),
            plex_info.get("parent_title"),
            media.get("summary"),
            plex_info.get("summary"),
            media.get("overview"),
            media.get("plot"),
            media.get("tagline"),
            plex_info.get("tagline"),
        )

        def _append_sequence(values: Any) -> None:
            for text in media_helpers._coerce_string_list(values):
                if text:
                    segments.append(text)

        for field in ("genres", "collections", "actors", "directors", "writers"):
            value = media.get(field)
            _append_sequence(value)
            if not value:
                _append_sequence(plex_info.get(field))

        deduped: list[str] = []
        seen: set[str] = set()
        for segment in segments:
            normalized = segment.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(normalized)
        return "\n".join(deduped)


async def _rerank_media_candidates(
        server: "PlexServer",
        query_text: str,
        items: list[AggregatedMediaItem],
        # Any is used here to accept arbitrary keyword arguments from tool callers
        # (title, actors, etc.) without duplicating the complex search model schema.
        **filters: Any,
    ) -> list[AggregatedMediaItem]:
        if len(items) <= 1:
            return items

        score_list: list[float] = [0.0] * len(items)
        if server.settings.use_reranker:
            reranker = await server.ensure_reranker()
            if reranker is not None:
                documents = [
                    _build_rerank_document(item) or str(item.get("title") or "")
                    for item in items
                ]
                try:
                    scores = await asyncio.to_thread(
                        reranker.predict,
                        [(query_text, document) for document in documents],
                    )
                    try:
                        score_list = [float(s) for s in list(scores)]
                    except TypeError:
                        score_list = [float(scores)]
                except Exception as exc:  # pragma: no cover
                    logger.warning("Failed to rerank media results with model: %s", exc)

        if len(score_list) != len(items):
            score_list = [0.0] * len(items)

        ranked: list[tuple[float, int, AggregatedMediaItem]] = []
        for idx, (item, model_score) in enumerate(zip(items, score_list)):
            score = model_score

            # Boost based on structured field matches if they were part of the query
            boost = 0.0

            # 1. Title matching (highest priority)
            q_title = filters.get("title")
            q_show_title = filters.get("show_title")

            item_title = str(item.get("title") or "").strip().lower()
            item_show_title = str(item.get("show_title") or "").strip().lower()

            if q_title:
                q_title_low = q_title.lower()
                ratio = fuzz.ratio(q_title_low, item_title)
                if ratio == 100:
                    boost += 10.0  # Exact title match
                elif ratio > 90:
                    boost += 5.0
                elif fuzz.token_set_ratio(q_title_low, item_title) == 100:
                    boost += 5.0
                elif fuzz.token_sort_ratio(q_title_low, item_title) > 95:
                    boost += 4.0

            if q_show_title:
                q_show_low = q_show_title.lower()
                if fuzz.ratio(q_show_low, item_show_title) == 100:
                    boost += 8.0
                elif fuzz.ratio(q_show_low, item_title) == 100:
                    boost += 5.0

            # 2. People matching
            for field in ("directors", "actors", "writers"):
                q_people = _listify(filters.get(field))
                if not q_people:
                    continue
                item_people = [
                    str(p).lower()
                    for p in media_helpers._coerce_string_list(item.get(field))
                ]
                for q_p in q_people:
                    q_p_low = q_p.lower()
                    if any(fuzz.ratio(q_p_low, i_p) > 90 for i_p in item_people):
                        boost += 3.0
                        break

            ranked.append((score + boost, idx, item))

        ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        return [entry[2] for entry in ranked]


def register_media_library_tools(server: "PlexServer") -> None:
    """Register media discovery tools and resources on the provided server."""

    def _media_tool(name: str, *, title: str, operation: str):
        return server.tool(
            name,
            title=title,
            meta={"category": "media-library", "operation": operation},
        )

    def _to_aggregated_models(
        payloads: list[AggregatedMediaItem],
    ) -> list[AggregatedMediaItemModel]:
        return [
            AggregatedMediaItemModel.model_validate(payload)
            if not isinstance(payload, AggregatedMediaItemModel)
            else payload
            for payload in payloads
        ]

    def _summarize(
        items: list[AggregatedMediaItem],
    ) -> MediaSummaryResponseModel:
        """Return the standard LLM summary for the provided media items."""

        summary = media_helpers.summarize_media_items_for_llm(items)
        return MediaSummaryResponseModel.model_validate(summary)

    @_media_tool("get-media", title="Get media details", operation="lookup")
    async def get_media(
        identifier: Annotated[
            str,
            Field(
                description="Rating key, IMDb/TMDb ID, or media title",
                examples=["49915", "tt8367814", "The Gentlemen"],
            ),
        ],
    ) -> list[AggregatedMediaItemModel]:
        """Retrieve media items by rating key, IMDb/TMDb ID or title."""

        cached_payload = media_helpers._get_cached_payload(server.cache, identifier)

        if cached_payload is not None:
            results = [cached_payload]
        else:
            # For Qdrant query, strip leading articles
            qdrant_identifier = _strip_leading_article(identifier) if identifier else identifier
            records = await media_helpers._find_records(server, qdrant_identifier, limit=10)
            results = [
                media_helpers._flatten_payload(
                    cast(Mapping[str, JSONValue] | None, r.payload)
                )
                for r in records
            ]
            if len(results) > 1 and not (
                identifier.isdigit() or identifier.startswith("tt")
            ):
                # Use the original identifier for reranking
                results = await _rerank_media_candidates(
                    server,
                    identifier,
                    results,
                    title=identifier
                )
        return _to_aggregated_models(results)

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
    ) -> MediaSummaryResponseModel:
        """Run a structured query against indexed payload fields and optional vector searches."""

        def _normalize_text(value: str | None) -> str | None:
            if value is None:
                return None
            text = value.strip()
            return text or None

        def _finalize(
            items: list[AggregatedMediaItem],
        ) -> MediaSummaryResponseModel:
            return _summarize(items)

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

        original_title_query = title
        title = _normalize_text(title)
        # For Qdrant query, strip leading articles
        qdrant_title = _strip_leading_article(title) if title else None
        show_title = _normalize_text(show_title)

        has_episode_hint = (
            season_number is not None or episode_number is not None
        )
        inferred_show_title: str | None = None
        if has_episode_hint and title and not show_title:
            records = await media_helpers._find_records(
                server, title, limit=5
            )
            for record in records:
                payload = cast(Mapping[str, JSONValue] | None, record.payload)
                flattened = media_helpers._flatten_payload(payload)
                show_candidate = flattened.get("show_title")
                if isinstance(show_candidate, str) and show_candidate.strip():
                    inferred_show_title = show_candidate
                    break
                plex_value = flattened.get("plex")
                if isinstance(plex_value, Mapping):
                    parent = plex_value.get("grandparent_title")
                    if isinstance(parent, str) and parent.strip():
                        inferred_show_title = parent
                        break
            if inferred_show_title:
                show_title = inferred_show_title
                title = None

        media_type = type
        if media_type is None and (
            show_title is not None or has_episode_hint
        ):
            media_type = "episode"

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

        rerank_query_text: str | None = None
        if not positive_point_ids:
            for candidate in (
                dense_query,
                sparse_query,
                original_title_query,
                summary,
                overview,
                plot,
                tagline,
            ):
                rerank_query_text = _normalize_text(candidate)
                if rerank_query_text:
                    break

        must: list[models.FieldCondition] = []
        keyword_prefetch_conditions: list[models.FieldCondition] = []

        if qdrant_title:
            must.append(
                models.FieldCondition(key="title", match=models.MatchText(text=qdrant_title))
            )
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

        try:
            res = await server.qdrant_client.query_points(
                collection_name="media-items",
                query=query_obj,
                using=using_param,
                prefetch=prefetch_param,
                query_filter=filter_obj,
                limit=limit,
                with_payload=True,
            )
        except ValueError as exc:
            if "Could not load model" not in str(exc):
                raise
            fallback_query = models.SampleQuery(sample=models.Sample.RANDOM)
            res = await server.qdrant_client.query_points(
                collection_name="media-items",
                query=fallback_query,
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
        if rerank_query_text:
            # Use the original title (with article) for reranking
            results = await _rerank_media_candidates(
                server,
                rerank_query_text,
                results,
                title=original_title_query,
                show_title=show_title,
                actors=actors,
                directors=directors,
                writers=writers,
                collections=collections,
                genres=genres,
            )
        return _finalize(results)

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
    ) -> MediaSummaryResponseModel:
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
        return _summarize(results)

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
    ) -> MediaSummaryResponseModel:
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
        return _summarize(results)

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
    ) -> MediaSummaryResponseModel:
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
        return _summarize(results)

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
