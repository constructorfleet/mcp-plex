"""Tests to guard Docker build artifacts stay optimized."""

import re
import tomllib
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet

DOCKERFILE = Path("Dockerfile")
DOCKERIGNORE = Path(".dockerignore")
CUDA_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"
BUILDER_STAGE_DESCRIPTOR = f"{CUDA_IMAGE} AS builder"
RUNTIME_STAGE_DESCRIPTOR = CUDA_IMAGE
DEPS_MANIFEST = Path("docker/pyproject.deps.toml")
PROJECT_MANIFEST = Path("pyproject.toml")
DOCKER_PYTHON_VERSION = "3.13"
UV_INSTALL_DIR = "/opt/uv"
UV_INSTALL_ENV_DIRECTIVE = f"ENV UV_PYTHON_INSTALL_DIR={UV_INSTALL_DIR}"
UV_INSTALL_COPY_DIRECTIVE = (
    f"COPY --from=builder --chown=app:app {UV_INSTALL_DIR} {UV_INSTALL_DIR}"
)


def _extract_stage(contents: str, stage_descriptor: str) -> str:
    """Return the body of a Docker stage defined by a FROM line."""

    segments = contents.split("FROM ")[1:]
    for segment in segments:
        header, *body_lines = segment.splitlines()
        if header.strip() == stage_descriptor:
            stage_body = "\n".join(body_lines)
            next_from = stage_body.find("\nFROM ")
            return stage_body if next_from == -1 else stage_body[:next_from]

    raise AssertionError(f"Missing Docker stage: {stage_descriptor}")



def _builder_section(contents: str) -> str:
    """Convenience accessor for the builder stage."""

    return _extract_stage(contents, BUILDER_STAGE_DESCRIPTOR)



def _runtime_section(contents: str) -> str:
    """Convenience accessor for the runtime stage."""

    return _extract_stage(contents, RUNTIME_STAGE_DESCRIPTOR)


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


def test_dockerfile_uses_nvidia_cuda_images(dockerfile_contents: str) -> None:
    """Ensure the Dockerfile relies on the CUDA runtime image for both stages."""

    assert f"FROM {BUILDER_STAGE_DESCRIPTOR}" in dockerfile_contents
    assert f"\nFROM {CUDA_IMAGE}" in dockerfile_contents


def test_dockerfile_does_not_install_curl_via_apt(dockerfile_contents: str) -> None:
    """The base image already supplies curl, so extra apt layers are wasteful."""

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

    runtime_section = _runtime_section(dockerfile_contents)
    expected_phrase = "COPY --from=builder /app/pyproject.toml ./pyproject.toml"

    assert (
        expected_phrase in runtime_section
    ), "Expected runtime stage to copy pyproject.toml for uv run entry points"


def test_builder_stage_copies_uv_binaries(dockerfile_contents: str) -> None:
    """Builder stage should source uv binaries from the official image."""

    builder_section = _builder_section(dockerfile_contents)

    assert (
        "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/"
        in builder_section
    ), "Builder stage must populate uv tools from the upstream image"


def test_builder_uv_sync_runs_in_non_editable_mode(dockerfile_contents: str) -> None:
    """Builder stage should follow uv guidance by avoiding editable installs."""

    builder_section = _builder_section(dockerfile_contents)

    sync_lines = [
        line.strip()
        for line in builder_section.splitlines()
        if "uv sync" in line
    ]

    assert sync_lines, "Expected builder stage to invoke uv sync"
    assert all(
        "--no-editable" in line for line in sync_lines
    ), "uv sync calls must disable editable installs in builder stage"
    assert any(
        "--no-install-project" in line for line in sync_lines
    ), "Builder stage should prime dependencies without the project using --no-install-project"


def test_builder_dependency_sync_pattern(dockerfile_contents: str) -> None:
    """Ensure the builder stage primes dependencies before adding the project."""

    builder_section = _builder_section(dockerfile_contents)

    first_sync_index = builder_section.find("uv sync --locked --no-install-project --no-editable")
    add_index = builder_section.find("ADD . /app")
    second_sync_index = builder_section.find("uv sync --locked --no-editable", add_index)

    assert first_sync_index != -1, "Dependency-only sync missing in builder stage"
    assert add_index != -1, "Expected builder stage to add project sources"
    assert second_sync_index != -1, "Expected builder stage to resync after adding project"
    assert first_sync_index < add_index < second_sync_index, "Sync and add ordering must follow uv guidance"


def test_runtime_stage_copies_virtualenv_with_chown(dockerfile_contents: str) -> None:
    """Ensure the runtime image receives the prepared virtual environment with ownership set."""

    runtime_section = _runtime_section(dockerfile_contents)

    assert (
        "COPY --from=builder --chown=app:app /app/.venv /app/.venv" in runtime_section
    ), "Runtime stage must copy the virtual environment with the app user ownership"

    assert (
        "ENTRYPOINT [\"./entrypoint.sh\"]" in runtime_section
    ), "Runtime stage should retain the project entrypoint"

    assert "CMD" in runtime_section, "Runtime stage should define a default command"


def test_dockerfile_preserves_uv_python_installation(dockerfile_contents: str) -> None:
    """Runtime stage should retain the uv-managed Python installation for app user access."""

    builder_section = _builder_section(dockerfile_contents)
    runtime_section = _runtime_section(dockerfile_contents)

    assert (
        UV_INSTALL_ENV_DIRECTIVE in builder_section
    ), "Builder stage must pin the uv Python install directory"
    assert (
        UV_INSTALL_ENV_DIRECTIVE in runtime_section
    ), "Runtime stage must expose the uv Python install directory"
    assert (
        UV_INSTALL_COPY_DIRECTIVE in runtime_section
    ), "Runtime stage should copy the uv-managed Python interpreter into place"


def _installed_python_version(contents: str) -> str:
    """Return the Python version the builder stage installs via uv."""

    match = re.search(r"uv python install (\S+)", _builder_section(contents))
    assert match, "Builder stage must install Python with uv"
    return match.group(1)


def test_builder_installs_supported_python(dockerfile_contents: str) -> None:
    """The image Python must be current and allowed by both project manifests."""

    installed = _installed_python_version(dockerfile_contents)

    assert installed == DOCKER_PYTHON_VERSION
    for manifest in (PROJECT_MANIFEST, DEPS_MANIFEST):
        requires = tomllib.loads(manifest.read_text(encoding="utf-8"))["project"][
            "requires-python"
        ]
        assert SpecifierSet(requires).contains(
            installed
        ), f"{manifest} requires-python {requires} excludes Python {installed}"


def test_installed_python_detects_unsupported_version() -> None:
    """An outdated or missing interpreter install should be caught."""

    stale = f"FROM {BUILDER_STAGE_DESCRIPTOR}\nRUN uv python install 3.12\n"
    assert _installed_python_version(stale) != DOCKER_PYTHON_VERSION
    assert not SpecifierSet(">=3.11,<3.13").contains(DOCKER_PYTHON_VERSION)

    with pytest.raises(AssertionError, match="must install Python"):
        _installed_python_version(f"FROM {BUILDER_STAGE_DESCRIPTOR}\nRUN uv sync\n")
