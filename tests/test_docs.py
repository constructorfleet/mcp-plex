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


def test_readme_documents_server_cache_and_reranker_settings():
    readme = Path(__file__).resolve().parent.parent / "README.md"
    content = readme.read_text(encoding="utf-8")

    for key in ("CACHE_SIZE", "USE_RERANKER", "PLEX_PLAYER_ALIASES"):
        assert key in content


@pytest.mark.parametrize("token", IMDB_RETRY_DOC_TOKENS)
def test_readme_documents_imdb_retry_controls(token: str) -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert token in readme
