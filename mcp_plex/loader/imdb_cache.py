from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class IMDbCache:
    """Simple persistent cache for IMDb API responses."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: Dict[str, Any]
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def get(self, imdb_id: str) -> dict[str, Any] | None:
        """Return cached data for ``imdb_id`` if present."""

        return self._data.get(imdb_id)

    def set(self, imdb_id: str, data: dict[str, Any]) -> None:
        """Store ``data`` under ``imdb_id`` and persist to disk."""

        self._data[imdb_id] = data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data))
