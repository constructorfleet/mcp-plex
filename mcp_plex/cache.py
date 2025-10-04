"""Compatibility shim for :mod:`mcp_plex.cache`."""

from __future__ import annotations

import warnings

from .common.cache import MediaCache

warnings.warn(
    "'mcp_plex.cache' is deprecated; import from 'mcp_plex.common.cache' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["MediaCache"]
