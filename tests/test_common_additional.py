from __future__ import annotations

from mcp_plex.common.text import slugify
from mcp_plex.common.types import TMDBEpisode
from mcp_plex.common.validation import coerce_plex_tag_id


class _BadInt:
    def __int__(self) -> int:
        raise TypeError("no int")


def test_coerce_plex_tag_id_handles_bad_objects():
    assert coerce_plex_tag_id(_BadInt()) == 0


def test_tmdb_episode_normalise_non_mapping():
    payload = ("value",)
    assert TMDBEpisode._normalise_episode_id(payload) is payload


def test_slugify_normalises_text():
    assert slugify("The Princess Bride") == "the-princess-bride"


def test_slugify_handles_diacritics_and_blanks():
    assert slugify("Amélie") == "amelie"
    assert slugify("   ") is None
