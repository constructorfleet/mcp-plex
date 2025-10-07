from __future__ import annotations

from mcp_plex.common.validation import coerce_plex_tag_id
from mcp_plex.common.types import TMDBEpisode


class _BadInt:
    def __int__(self) -> int:
        raise TypeError("no int")


def test_coerce_plex_tag_id_handles_bad_objects():
    assert coerce_plex_tag_id(_BadInt()) == 0


def test_tmdb_episode_normalise_non_mapping():
    payload = ("value",)
    assert TMDBEpisode._normalise_episode_id(payload) is payload
