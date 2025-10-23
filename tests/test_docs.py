from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


IMDB_RETRY_DOC_TOKENS = (
    "--imdb-cache",
    "IMDB_CACHE",
    "--imdb-max-retries",
    "IMDB_MAX_RETRIES",
    "--imdb-backoff",
    "IMDB_BACKOFF",
    "--imdb-requests-per-window",
    "IMDB_REQUESTS_PER_WINDOW",
    "--imdb-window-seconds",
    "IMDB_WINDOW_SECONDS",
    "--imdb-queue",
    "IMDB_QUEUE",
    "Aggressive retry policy",
    "Conservative retry policy",
)


def test_readme_documents_server_cache_and_reranker_settings():
    content = read_readme()

    for key in ("CACHE_SIZE", "USE_RERANKER", "PLEX_PLAYER_ALIASES"):
        assert key in content


@pytest.mark.parametrize("token", IMDB_RETRY_DOC_TOKENS)
def test_readme_documents_imdb_retry_controls(token: str) -> None:
    readme = read_readme()
    assert token in readme


def test_readme_documents_dev_checks() -> None:
    readme = read_readme()

    assert "uv run ruff check ." in readme
    assert "uv run pytest" in readme


def read_readme() -> str:
    return (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
