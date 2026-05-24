"""Tests for Docker entrypoint secret resolution."""

from __future__ import annotations

import os
import subprocess
from textwrap import dedent
from pathlib import Path


ENTRYPOINT = Path("entrypoint.sh").resolve()


def _run_entrypoint(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    process_env.update(env)
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *args],
        check=True,
        capture_output=True,
        text=True,
        env=process_env,
    )


def test_entrypoint_resolves_whitelisted_secret_files(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_server = fake_bin / "mcp-server"
    fake_server.write_text(
        dedent(
            """\
            #!/usr/bin/env python3
            import os
            print(os.getenv('PLEX_TOKEN'))
            print(os.getenv('TMDB_API_KEY'))
            print(os.getenv('QDRANT_API_KEY'))
            print(os.getenv('PLEX_TOKEN_FILE'))
            print(os.getenv('PLEX_CLIENTS_FILE'))
            """
        ),
        encoding="utf-8",
    )
    fake_server.chmod(0o755)

    plex_token_file = tmp_path / "plex_token"
    tmdb_api_key_file = tmp_path / "tmdb_api_key"
    qdrant_api_key_file = tmp_path / "qdrant_api_key"
    plex_token_file.write_text("plex-secret\n", encoding="utf-8")
    tmdb_api_key_file.write_text("tmdb-secret\n", encoding="utf-8")
    qdrant_api_key_file.write_text("qdrant-secret\n", encoding="utf-8")

    result = _run_entrypoint(
        "mcp-server",
        env={
            "PATH": f"{fake_bin}:{os.environ.get('PATH', '')}",
            "PLEX_TOKEN_FILE": str(plex_token_file),
            "TMDB_API_KEY_FILE": str(tmdb_api_key_file),
            "QDRANT_API_KEY_FILE": str(qdrant_api_key_file),
            "PLEX_CLIENTS_FILE": "/run/secrets/clients.yaml",
        },
    )

    assert result.stdout.splitlines() == [
        "plex-secret",
        "tmdb-secret",
        "qdrant-secret",
        "None",
        "/run/secrets/clients.yaml",
    ]


def test_entrypoint_fails_when_secret_file_is_missing(tmp_path: Path) -> None:
    missing_secret = tmp_path / "missing"

    result = subprocess.run(
        ["bash", str(ENTRYPOINT), "python", "-c", "print('should not run')"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": os.environ.get("PATH", ""),
            "PLEX_TOKEN_FILE": str(missing_secret),
        },
    )

    assert result.returncode != 0
    assert "is not readable" in result.stderr
