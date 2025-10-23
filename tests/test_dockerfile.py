"""Tests for Dockerfile configuration related to uv."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"


def read_dockerfile_contents() -> str:
    """Load the Dockerfile contents for reuse across tests."""

    return DOCKERFILE_PATH.read_text(encoding="utf-8")


def test_uv_binary_path_is_added_before_usage() -> None:
    """Ensure the Dockerfile exposes uv on PATH before it's invoked."""

    dockerfile_contents = read_dockerfile_contents()

    expected_line = 'ENV PATH="/root/.local/bin:$PATH"'

    assert (
        expected_line in dockerfile_contents
    ), "Dockerfile must expose uv installation directory before using uv"


def test_uv_binary_is_copied_into_usr_local_bin() -> None:
    """Ensure uv binaries are copied into a globally accessible location."""

    dockerfile_contents = read_dockerfile_contents()

    expected_instruction = (
        "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/"
    )

    assert (
        expected_instruction in dockerfile_contents
    ), "Dockerfile must copy uv binaries into /usr/local/bin"
