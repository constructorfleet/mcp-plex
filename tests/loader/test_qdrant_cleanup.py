import asyncio
from types import SimpleNamespace

from mcp_plex.loader import qdrant as qdrant_module


class CleanupClient:
    def __init__(self, batches: list[list[SimpleNamespace]]) -> None:
        self._batches = list(batches)
        self.scroll_calls = 0
        self.deleted: list[list[int | str]] = []

    async def scroll(
        self, *, offset: None | dict[str, int] = None, **_: object
    ) -> tuple[list[SimpleNamespace], None | dict[str, int]]:
        if self.scroll_calls >= len(self._batches):
            return [], None
        batch = self._batches[self.scroll_calls]
        self.scroll_calls += 1
        offset = None if self.scroll_calls == len(self._batches) else {"page": self.scroll_calls}
        return batch, offset

    async def delete(self, *, points_selector, **_: object) -> None:
        self.deleted.append(list(points_selector.points))


def _record(record_id: int | str, rating_key: str | None) -> SimpleNamespace:
    payload = {"data": {"plex": {"rating_key": rating_key}}} if rating_key is not None else {}
    return SimpleNamespace(id=record_id, payload=payload)


def test_delete_missing_rating_keys_removes_stale_points():
    client = CleanupClient(
        [
            [_record("1", "1"), _record("2", "2")],
            [_record("3", "4"), _record("4", None)],
        ]
    )

    deleted, scanned = asyncio.run(
        qdrant_module._delete_missing_rating_keys(
            client,
            collection_name="media-items",
            active_rating_keys={"1", "4"},
            scroll_limit=2,
        )
    )

    assert deleted == 2
    assert scanned == 4
    assert client.deleted == [["2"], ["4"]]
    assert client.scroll_calls == 2


def test_delete_missing_rating_keys_skips_when_no_keys():
    client = CleanupClient([])

    deleted, scanned = asyncio.run(
        qdrant_module._delete_missing_rating_keys(
            client,
            collection_name="media-items",
            active_rating_keys=set(),
        )
    )

    assert deleted == 0
    assert scanned == 0
    assert client.scroll_calls == 0
    assert client.deleted == []


class LoopingCleanupClient:
    """Simulate Qdrant returning the same page whenever the offset is ``None``."""

    def __init__(self, batch: list[SimpleNamespace]) -> None:
        self.batch = batch
        self.scroll_calls = 0
        self.deleted: list[list[int | str]] = []

    async def scroll(self, **_: object) -> tuple[list[SimpleNamespace], None]:
        self.scroll_calls += 1
        # Always return the same data with a null offset to mimic the Qdrant behavior
        # described in the bug report.
        return self.batch, None

    async def delete(self, *, points_selector, **_: object) -> None:
        self.deleted.append(list(points_selector.points))


def test_delete_missing_rating_keys_breaks_when_offset_none():
    client = LoopingCleanupClient([_record("1", "stale"), _record("2", "2")])

    deleted, scanned = asyncio.run(
        qdrant_module._delete_missing_rating_keys(
            client,
            collection_name="media-items",
            active_rating_keys={"2"},
            scroll_limit=10,
        )
    )

    assert deleted == 1
    assert scanned == 2
    assert client.deleted == [["1"]]
    # Ensures we only read the looping page once instead of spinning forever.
    assert client.scroll_calls == 1
