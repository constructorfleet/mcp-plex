"""Compatibility shim for :mod:`mcp_plex.types`."""

from __future__ import annotations

import warnings

from .common import types as _types

warnings.warn(
    "'mcp_plex.types' is deprecated; import from 'mcp_plex.common.types' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(_types.__all__)

globals().update({name: getattr(_types, name) for name in __all__})
