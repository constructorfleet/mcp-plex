"""Validation helpers shared across packages."""

from __future__ import annotations

import asyncio
from typing import Callable, SupportsInt, TypeVar, Union, Awaitable

T = TypeVar("T")


def require_positive(value: int, *, name: str) -> int:
    """Return *value* if it is a positive integer, otherwise raise an error."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def coerce_plex_tag_id(raw_id: int | str | SupportsInt | None) -> int:
    """Best-effort conversion of Plex media tag identifiers to integers."""

    if raw_id is None:
        return 0
    if isinstance(raw_id, bool):
        return int(raw_id)
    if isinstance(raw_id, int):
        return raw_id
    if isinstance(raw_id, str):
        raw_id = raw_id.strip()
        if not raw_id:
            return 0
        try:
            return int(raw_id)
        except ValueError:
            return 0
    try:
        return int(raw_id)
    except (TypeError, ValueError):
        return 0


def retry(
    retries: int = 3, delay: float = 1.0, backoff: float = 2.0
) -> Callable[[Callable[..., Union[T, Awaitable[T]]]], Callable[..., Awaitable[T]]]:
    """Retry decorator for handling transient errors."""

    def decorator(func: Callable[..., Union[T, Awaitable[T]]]) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args, **kwargs) -> T:
            attempt = 0
            current_delay = delay

            while attempt < retries:
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)  # type: ignore
                    return func(*args, **kwargs)  # type: ignore
                except Exception as e:
                    attempt += 1
                    if attempt >= retries:
                        raise
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff

            raise RuntimeError("Retry decorator exhausted all attempts without success.")

        return wrapper

    return decorator


__all__ = ["require_positive", "coerce_plex_tag_id", "retry"]
