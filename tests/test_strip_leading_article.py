from mcp_plex.server.tools.media_library import _strip_leading_article
import pytest

@pytest.mark.parametrize("title,expected", [
    ("The Predator", "Predator"),
    ("A Beautiful Mind", "Beautiful Mind"),
    ("An Education", "Education"),
    ("the Matrix", "Matrix"),
    ("a Quiet Place", "Quiet Place"),
    ("an Honest Liar", "Honest Liar"),
    ("Predator", "Predator"),
    ("The   ", "The"),
    ("", ""),
    (None, None),
])
def test_strip_leading_article(title, expected):
    assert _strip_leading_article(title) == expected
