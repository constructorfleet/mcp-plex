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
