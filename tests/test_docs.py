from pathlib import Path
import pytest


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



def _read_readme_text() -> str:
    return Path("README.md").read_text(encoding="utf-8")


def test_readme_documents_server_cache_and_reranker_settings():
    content = _read_readme_text()

    for key in ("CACHE_SIZE", "USE_RERANKER", "PLEX_PLAYER_ALIASES"):
        assert key in content


@pytest.mark.parametrize("token", IMDB_RETRY_DOC_TOKENS)
def test_readme_documents_imdb_retry_controls(token: str) -> None:
    content = _read_readme_text()
    assert token in content


def test_readme_documents_ruff_check_command() -> None:
    content = _read_readme_text()
    assert "uv run ruff check ." in content
