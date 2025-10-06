import pytest

from mcp_plex.common.validation import require_positive


def test_require_positive_accepts_positive_int():
    assert require_positive(5, name="value") == 5


@pytest.mark.parametrize("bad", [0, -1, -100])
def test_require_positive_rejects_non_positive_int(bad):
    with pytest.raises(ValueError, match="value must be positive"):
        require_positive(bad, name="value")


@pytest.mark.parametrize("bad_type", [1.5, "1", None, object(), True])
def test_require_positive_enforces_int_type(bad_type):
    with pytest.raises(TypeError, match="value must be an int"):
        require_positive(bad_type, name="value")  # type: ignore[arg-type]
