from __future__ import annotations

import asyncio
from collections import deque
from typing import Iterable


class IMDbRetryQueue(asyncio.Queue[str]):
    """Queue that tracks items in a deque for safe serialization."""

    def __init__(self, initial: Iterable[str] | None = None):
        super().__init__()
        self._items: deque[str] = deque()
        if initial:
            for imdb_id in initial:
                imdb_id_str = str(imdb_id)
                super().put_nowait(imdb_id_str)
                self._items.append(imdb_id_str)

    def put_nowait(self, item: str) -> None:  # type: ignore[override]
        super().put_nowait(item)
        self._items.append(item)

    def get_nowait(self) -> str:  # type: ignore[override]
        if not self._items:
            raise RuntimeError("Desynchronization: Queue is not empty but self._items is empty.")
        try:
            item = super().get_nowait()
        except asyncio.QueueEmpty:
            raise RuntimeError(
                "Desynchronization: self._items is not empty but asyncio.Queue is empty."
            )
        self._items.popleft()
        return item

    def snapshot(self) -> list[str]:
        """Return a list of the current queue contents."""

        return list(self._items)


__all__ = ["IMDbRetryQueue"]
