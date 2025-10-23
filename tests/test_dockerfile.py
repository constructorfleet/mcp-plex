"""Tests for Dockerfile configuration related to uv."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"


def test_uv_binary_path_is_added_before_usage() -> None:
    """Ensure the Dockerfile exposes uv on PATH before it's invoked."""

    dockerfile_contents = DOCKERFILE_PATH.read_text(encoding="utf-8")

    expected_line = 'ENV PATH="/root/.local/bin:$PATH"'

    assert (
        expected_line in dockerfile_contents
    ), "Dockerfile must expose uv installation directory before using uv"
