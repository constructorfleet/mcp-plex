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


class _SupportsInt:
    def __int__(self) -> int:
        return 128


@pytest.mark.parametrize(
    "raw, expected",
    [
        (7, 7),
        (True, 1),
        (" 42 ", 42),
        (_SupportsInt(), 128),
    ],
)
def test_coerce_plex_tag_id_normalizes_values(raw, expected) -> None:
    assert coerce_plex_tag_id(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "not-a-number"])
def test_coerce_plex_tag_id_handles_invalid_values(raw) -> None:
    assert coerce_plex_tag_id(raw) == 0
