"""Tests for shared validation helpers."""

import pytest

from mcp_plex.common.validation import coerce_plex_tag_id, require_positive


def test_require_positive_accepts_positive_int() -> None:
    assert require_positive(5, name="value") == 5


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_require_positive_rejects_non_positive_int(bad: int) -> None:
    with pytest.raises(ValueError, match="value must be positive"):
        require_positive(bad, name="value")


@pytest.mark.parametrize("bad_type", [1.5, "1", None, object(), True])
def test_require_positive_enforces_int_type(bad_type: object) -> None:
    with pytest.raises(TypeError, match="value must be an int"):
        require_positive(bad_type, name="value")  # type: ignore[arg-type]


def test_coerce_plex_tag_id_accepts_ints() -> None:
    assert coerce_plex_tag_id(7) == 7


def test_coerce_plex_tag_id_coerces_strings() -> None:
    assert coerce_plex_tag_id(" 42 ") == 42


def test_coerce_plex_tag_id_handles_invalid_values() -> None:
    assert coerce_plex_tag_id(None) == 0
    assert coerce_plex_tag_id("not-a-number") == 0
