"""In-memory LRU cache for media payload and artwork data."""

from __future__ import annotations

from collections import OrderedDict
from typing import TypeVar

from .types import JSONValue


CachedPayload = dict[str, JSONValue]
_CacheValueT = TypeVar("_CacheValueT")


class MediaCache:
    """LRU caches for media payload, posters, and backgrounds."""

    def __init__(self, size: int = 128) -> None:
        self.size = size
        self._payload: OrderedDict[str, CachedPayload] = OrderedDict()
        self._poster: OrderedDict[str, str] = OrderedDict()
        self._background: OrderedDict[str, str] = OrderedDict()

    def _set(
        self, cache: OrderedDict[str, _CacheValueT], key: str, value: _CacheValueT
    ) -> None:
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        while len(cache) > self.size:
            cache.popitem(last=False)

    def _get(
        self, cache: OrderedDict[str, _CacheValueT], key: str
    ) -> _CacheValueT | None:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def get_payload(self, key: str) -> CachedPayload | None:
        return self._get(self._payload, key)

    def set_payload(self, key: str, value: CachedPayload) -> None:
        self._set(self._payload, key, value)

    def get_poster(self, key: str) -> str | None:
        return self._get(self._poster, key)

    def set_poster(self, key: str, value: str) -> None:
        self._set(self._poster, key, value)

    def get_background(self, key: str) -> str | None:
        return self._get(self._background, key)

    def set_background(self, key: str, value: str) -> None:
        self._set(self._background, key, value)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._payload.clear()
        self._poster.clear()
        self._background.clear()
