"""Validation helpers shared across packages."""

from __future__ import annotations


def require_positive(value: int, *, name: str) -> int:
    """Return *value* if it is a positive integer, otherwise raise an error."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


__all__ = ["require_positive"]
