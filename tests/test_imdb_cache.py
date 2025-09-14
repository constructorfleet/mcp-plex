import json
from pathlib import Path

from mcp_plex.imdb_cache import IMDbCache


def test_imdb_cache_loads_existing_and_persists(tmp_path: Path):
    path = tmp_path / "cache.json"
    path.write_text(
        json.dumps(
            {
                "tt0111161": {
                    "id": "tt0111161",
                    "primaryTitle": "The Shawshank Redemption",
                }
            }
        )
    )
    cache = IMDbCache(path)
    assert cache.get("tt0111161") == {
        "id": "tt0111161",
        "primaryTitle": "The Shawshank Redemption",
    }

    cache.set(
        "tt0068646", {"id": "tt0068646", "primaryTitle": "The Godfather"}
    )
    assert json.loads(path.read_text()) == {
        "tt0111161": {
            "id": "tt0111161",
            "primaryTitle": "The Shawshank Redemption",
        },
        "tt0068646": {
            "id": "tt0068646",
            "primaryTitle": "The Godfather",
        },
    }


def test_imdb_cache_invalid_file(tmp_path: Path):
    path = tmp_path / "cache.json"
    path.write_text("not json")
    cache = IMDbCache(path)
    assert cache.get("tt0111161") is None
    cache.set("tt0111161", {"id": "tt0111161"})
    assert cache.get("tt0111161") == {"id": "tt0111161"}
