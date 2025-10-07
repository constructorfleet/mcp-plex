from __future__ import annotations

from mcp_plex.server import media as media_helpers


def test_flatten_payload_with_none():
    assert media_helpers._flatten_payload(None) == {}


def test_normalize_identifier_handles_bad_string():
    class BadStr:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    assert media_helpers._normalize_identifier(BadStr()) is None


def test_extract_plex_metadata_returns_empty_when_missing():
    assert media_helpers._extract_plex_metadata({"plex": "not-a-dict"}) == {}


def test_summarize_media_items_for_llm_handles_empty_collection():
    summary = media_helpers.summarize_media_items_for_llm([])
    assert summary == {"total_results": 0, "results": []}


def test_summarize_media_items_for_llm_returns_expected_fields():
    item = {
        "title": "The Gentlemen",
        "year": 2019,
        "type": "movie",
        "summary": "A crime lord tries to sell his empire.",
        "tagline": "Criminal. Class.",
        "genres": ["Action", "Comedy"],
        "collections": ["Crime Capers"],
        "actors": [{"tag": "Matthew McConaughey"}, "Charlie Hunnam"],
        "directors": ["Guy Ritchie"],
        "writers": ["Guy Ritchie"],
        "imdb": {"id": "tt8367814"},
        "tmdb": {"id": 568467},
        "plex": {"rating_key": "49915", "title": "The Gentlemen"},
    }
    summary = media_helpers.summarize_media_items_for_llm([item])
    assert summary["total_results"] == 1
    entry = summary["results"][0]
    assert entry["title"] == "The Gentlemen"
    assert entry["identifiers"]["rating_key"] == "49915"
    assert "Matthew McConaughey" in entry["actors"]
    assert "Guy Ritchie" in entry["directors"]
    assert "Crime Capers" in entry["collections"]
    assert "Criminal. Class." in entry["description"]
