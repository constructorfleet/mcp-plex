"""mcp-plex package."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'mcp_plex\\.loader' found in sys.modules after import of package 'mcp_plex'.*",
    category=RuntimeWarning,
)

__all__ = []
