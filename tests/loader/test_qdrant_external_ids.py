import asyncio

from qdrant_client import models

from mcp_plex.loader import qdrant as qdrant_module


class DummyQdrantClient:
    def __init__(self) -> None:
        self.scroll_calls: list[dict[str, object]] = []

    async def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: models.Filter,
        limit: int,
        with_payload: bool,
        with_vectors: bool,
    ) -> tuple[list[models.Record], None]:
        self.scroll_calls.append(
            {
                "collection_name": collection_name,
                "scroll_filter": scroll_filter,
                "limit": limit,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        return ([], None)


def test_find_record_by_external_ids_requires_all_ids():
    client = DummyQdrantClient()

    async def run_test() -> None:
        await qdrant_module._find_record_by_external_ids(
            client,
            "collection",
            imdb_id="tt123",
            tmdb_id="456",
            plex_guid="plex://movie/1",
        )

    asyncio.run(run_test())

    assert len(client.scroll_calls) == 1
    scroll_filter = client.scroll_calls[0]["scroll_filter"]
    expected = models.Filter(
        must=[
            models.FieldCondition(
                key="data.imdb.id",
                match=models.MatchValue(value="tt123"),
            ),
            models.FieldCondition(
                key="data.tmdb.id",
                match=models.MatchValue(value=456),
            ),
            models.FieldCondition(
                key="data.plex.guid",
                match=models.MatchValue(value="plex://movie/1"),
            ),
        ]
    )
    assert scroll_filter.model_dump() == expected.model_dump()
