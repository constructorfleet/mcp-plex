"""Compatibility shim for :mod:`mcp_plex.imdb_cache`."""

from __future__ import annotations

import warnings

from .loader.imdb_cache import IMDbCache

warnings.warn(
    "'mcp_plex.imdb_cache' is deprecated; import from 'mcp_plex.loader.imdb_cache' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["IMDbCache"]
