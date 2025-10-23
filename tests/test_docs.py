from pathlib import Path
import pytest


def read_readme() -> str:
    return Path("README.md").read_text(encoding="utf-8")


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


def test_readme_documents_live_plex_configuration_requirements() -> None:
    readme = read_readme()

    required_phrases = (
        "### Configure live Plex access",
        "--plex-url",
        "--plex-token",
        "--tmdb-api-key",
        "PLEX_URL",
        "PLEX_TOKEN",
        "TMDB_API_KEY",
    )

    for phrase in required_phrases:
        assert phrase in readme


@pytest.mark.parametrize("token", IMDB_RETRY_DOC_TOKENS)
def test_readme_documents_imdb_retry_controls(token: str) -> None:
    readme = read_readme()
    assert token in readme
