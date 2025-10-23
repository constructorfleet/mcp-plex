"""Tests to guard Docker build artifacts stay optimized."""

from pathlib import Path

import pytest


DOCKERFILE = Path("Dockerfile")
DOCKERIGNORE = Path(".dockerignore")


@pytest.fixture()
def dockerfile_contents() -> str:
    """Read the Dockerfile once so individual checks stay lean."""

    return DOCKERFILE.read_text(encoding="utf-8")


@pytest.fixture()
def dockerignore_entries() -> set[str]:
    """Provide the normalized set of .dockerignore entries for assertions."""

    assert DOCKERIGNORE.exists(), "Expected .dockerignore to prevent large contexts"
    return {
        line.strip()
        for line in DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def test_dockerfile_uses_runtime_cuda_image(dockerfile_contents: str) -> None:
    """Ensure the Dockerfile relies on the lighter runtime CUDA base image."""

    assert "FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04" in dockerfile_contents


def test_dockerfile_does_not_install_curl_via_apt(dockerfile_contents: str) -> None:
    """The base CUDA image already supplies curl, so extra apt layers are wasteful."""

    assert "apt-get install" not in dockerfile_contents


def test_dockerfile_avoids_copying_entire_context(dockerfile_contents: str) -> None:
    """Guard against COPY . . which bloats the final image size."""

    assert "COPY . ." not in dockerfile_contents


def test_dockerignore_excludes_heavy_paths(dockerignore_entries: set[str]) -> None:
    """Ensure bulky development directories stay out of the Docker build context."""

    required_patterns = {"tests/", "docs/", "sample-data/", ".git", ".venv", "mcp_plex.egg-info/"}

    for pattern in required_patterns:
        assert pattern in dockerignore_entries, f"Missing {pattern} from .dockerignore"
