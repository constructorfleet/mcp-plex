import asyncio

import pytest

from mcp_plex.loader import IMDbSource, PlexSource, TMDBSource


@pytest.mark.parametrize(
    ("source_cls", "expected_source"),
    [
        (PlexSource, "plex"),
        (TMDBSource, "tmdb"),
        (IMDbSource, "imdb"),
    ],
)
def test_data_source_fetch_returns_structured_payload(source_cls, expected_source) -> None:
    source = source_cls()
    payload = asyncio.run(source.fetch_data())

    assert payload["source"] == expected_source
    assert isinstance(payload["items"], list)


@pytest.mark.parametrize(
    ("source_cls", "expected_source"),
    [
        (PlexSource, "plex"),
        (TMDBSource, "tmdb"),
        (IMDbSource, "imdb"),
    ],
)
def test_data_source_validate_accepts_valid_payload(source_cls, expected_source) -> None:
    source = source_cls()
    payload = {"source": expected_source, "items": [{"title": "Sample Title", "year": 1999}]}

    assert source.validate(payload)


@pytest.mark.parametrize(
    ("source_cls", "expected_source"),
    [
        (PlexSource, "plex"),
        (TMDBSource, "tmdb"),
        (IMDbSource, "imdb"),
    ],
)
def test_data_source_validate_rejects_invalid_payload(source_cls, expected_source) -> None:
    source = source_cls()

    assert not source.validate({"source": "other", "items": []})
    assert not source.validate({"source": expected_source, "items": "bad"})
