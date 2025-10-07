from __future__ import annotations

import json

from mcp_plex.loader.imdb_cache import IMDbCache


def test_imdb_cache_accepts_non_dict_payload(tmp_path):
    cache_path = tmp_path / "imdb.json"
    cache_path.write_text(json.dumps({"tt123": ["raw"]}))

    cache = IMDbCache(cache_path)

    assert cache.get("tt123") == ["raw"]


def test_imdb_cache_warns_on_non_object(tmp_path, caplog):
    cache_path = tmp_path / "imdb.json"
    cache_path.write_text(json.dumps([1, 2, 3]))

    IMDbCache(cache_path)

    assert "did not contain an object" in caplog.text
