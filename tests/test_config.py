import pytest
from pydantic import ValidationError

from mcp_plex.config import Settings


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "7001")
    settings = Settings()
    assert settings.qdrant_port == 7001


def test_settings_invalid_cache_size(monkeypatch):
    monkeypatch.setenv("CACHE_SIZE", "notint")
    with pytest.raises(ValidationError):
        Settings()
