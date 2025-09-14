import json
from pathlib import Path

from mcp_plex.imdb_cache import IMDbCache


def test_imdb_cache_loads_existing_and_persists(tmp_path: Path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"tt1": {"id": "tt1"}}))
    cache = IMDbCache(path)
    assert cache.get("tt1") == {"id": "tt1"}

    cache.set("tt2", {"id": "tt2"})
    assert json.loads(path.read_text()) == {
        "tt1": {"id": "tt1"},
        "tt2": {"id": "tt2"},
    }


def test_imdb_cache_invalid_file(tmp_path: Path):
    path = tmp_path / "cache.json"
    path.write_text("not json")
    cache = IMDbCache(path)
    assert cache.get("tt1") is None
    cache.set("tt1", {"id": "tt1"})
    assert cache.get("tt1") == {"id": "tt1"}
