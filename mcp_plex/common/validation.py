"""Validation helpers shared across packages."""

from __future__ import annotations

from typing import Any


def require_positive(value: int, *, name: str) -> int:
    """Return *value* if it is a positive integer, otherwise raise an error."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def coerce_plex_tag_id(raw_id: Any) -> int:
    """Best-effort conversion of Plex media tag identifiers to integers."""

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


__all__ = ["require_positive", "coerce_plex_tag_id"]
