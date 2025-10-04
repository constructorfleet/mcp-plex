"""Compatibility shim for :mod:`mcp_plex.config`."""

from __future__ import annotations

import warnings

from .server.config import Settings

warnings.warn(
    "'mcp_plex.config' is deprecated; import from 'mcp_plex.server.config' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Settings"]
