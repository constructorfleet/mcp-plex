from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypeAlias, cast

from pydantic import ValidationError

from ..common.types import IMDbTitle, JSONValue


CachedIMDbPayload: TypeAlias = IMDbTitle | JSONValue


class IMDbCache:
    """Simple persistent cache for IMDb API responses."""

    _logger = logging.getLogger(__name__)

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, CachedIMDbPayload] = {}
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
                    loaded = json.loads(raw_contents)
                except (json.JSONDecodeError, UnicodeError) as exc:
                    self._logger.warning(
                        "Failed to decode IMDb cache JSON from %s; starting with empty cache.",
                        path,
                        exc_info=exc,
                    )
                else:
                    if isinstance(loaded, dict):
                        hydrated: dict[str, CachedIMDbPayload] = {}
                        for key, value in loaded.items():
                            imdb_id = str(key)
                            payload: CachedIMDbPayload
                            if isinstance(value, dict):
                                try:
                                    payload = IMDbTitle.model_validate(value)
                                except ValidationError as exc:
                                    self._logger.debug(
                                        "Failed to validate cached IMDb payload for %s; falling back to raw JSON.",
                                        imdb_id,
                                        exc_info=exc,
                                    )
                                    payload = cast(JSONValue, value)
                            else:
                                payload = cast(JSONValue, value)
                            hydrated[imdb_id] = payload
                        self._data = hydrated
                    else:
                        self._logger.warning(
                            "IMDb cache at %s did not contain an object; ignoring its contents.",
                            path,
                        )

    def get(self, imdb_id: str) -> CachedIMDbPayload | None:
        """Return cached data for ``imdb_id`` if present."""

        return self._data.get(imdb_id)

    def set(self, imdb_id: str, data: CachedIMDbPayload) -> None:
        """Store ``data`` under ``imdb_id`` and persist to disk."""

        self._data[imdb_id] = data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {
            key: value.model_dump() if isinstance(value, IMDbTitle) else value
            for key, value in self._data.items()
        }
        self.path.write_text(json.dumps(serialisable))
