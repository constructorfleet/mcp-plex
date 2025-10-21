from pathlib import Path


def test_readme_documents_server_cache_and_reranker_settings():
    readme = Path(__file__).resolve().parent.parent / "README.md"
    content = readme.read_text(encoding="utf-8")

    for key in ("CACHE_SIZE", "USE_RERANKER", "PLEX_PLAYER_ALIASES"):
        assert key in content
