import types
from pathlib import Path

from mcp_plex.loader import _extract_external_ids, _load_from_sample


def test_extract_external_ids():
    guid_objs = [types.SimpleNamespace(id="imdb://tt123"), types.SimpleNamespace(id="tmdb://456")]
    item = types.SimpleNamespace(guids=guid_objs)
    ids = _extract_external_ids(item)
    assert ids.imdb == "tt123"
    assert ids.tmdb == "456"


def test_load_from_sample_returns_items():
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    items = _load_from_sample(sample_dir)
    assert len(items) == 2
    assert {i.plex.type for i in items} == {"movie", "episode"}
