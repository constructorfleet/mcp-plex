from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


class IMDbCache:
    """Simple persistent cache for IMDb API responses."""

    _logger = logging.getLogger(__name__)

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = {}
        if path.exists():
            try:
                raw_contents = path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - ensure any read failure is surfaced
                self._logger.warning(
                    "Failed to read IMDb cache from %s; starting with empty cache.",
                    path,
                    exc_info=exc,
                )
            else:
                try:
                    self._data = json.loads(raw_contents)
                except (json.JSONDecodeError, UnicodeError) as exc:
                    self._logger.warning(
                        "Failed to decode IMDb cache JSON from %s; starting with empty cache.",
                        path,
                        exc_info=exc,
                    )

    def get(self, imdb_id: str) -> dict[str, Any] | None:
        """Return cached data for ``imdb_id`` if present."""

        return self._data.get(imdb_id)

    def set(self, imdb_id: str, data: dict[str, Any]) -> None:
        """Store ``data`` under ``imdb_id`` and persist to disk."""

        self._data[imdb_id] = data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data))
