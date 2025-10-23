"""Tests to guard Docker build artifacts stay optimized."""

from pathlib import Path

import pytest


DOCKERFILE = Path("Dockerfile")
DOCKERIGNORE = Path(".dockerignore")
BASE_STAGE_DESCRIPTOR = "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base"


def _extract_stage(contents: str, stage_descriptor: str) -> str:
    """Return the body of a Docker stage defined by a FROM line."""

    marker = f"FROM {stage_descriptor}"
    assert marker in contents, f"Missing Docker stage: {stage_descriptor}"

    stage_body = contents.split(marker, 1)[1]
    next_from = stage_body.find("\nFROM ")
    return stage_body if next_from == -1 else stage_body[:next_from]


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


def test_dockerfile_copies_project_metadata_for_uv(dockerfile_contents: str) -> None:
    """The runtime image must include pyproject metadata so uv can resolve scripts."""

    expected_phrase = "COPY pyproject.toml"

    assert (
        expected_phrase in dockerfile_contents
    ), "Expected runtime stage to copy pyproject.toml for uv run entry points"


def test_dockerfile_sets_uv_path_in_builder_stage(dockerfile_contents: str) -> None:
    """Ensure the builder stage exports the uv binary on PATH before syncing deps."""

    builder_section = _extract_stage(dockerfile_contents, "base AS builder")
    expected_env = 'ENV PATH="/root/.local/bin:$PATH"'

    assert (
        expected_env in builder_section
    ), "Builder stage must export uv install path before invoking uv"


def test_dockerfile_sets_uv_install_dir_for_base(dockerfile_contents: str) -> None:
    """Ensure the base stage installs uv to a globally accessible directory."""

    base_section = _extract_stage(dockerfile_contents, BASE_STAGE_DESCRIPTOR)
    expected_env = 'ENV XDG_BIN_HOME="/usr/local/bin"'

    assert expected_env in base_section, "Base stage must direct uv installer to /usr/local/bin"
