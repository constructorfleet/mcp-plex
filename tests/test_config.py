import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from mcp_plex.config import Settings


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("QDRANT_PORT", "7001")
    settings = Settings()
    assert settings.qdrant_port == 7001


def test_settings_invalid_cache_size(monkeypatch):
    monkeypatch.setenv("CACHE_SIZE", "notint")
    with pytest.raises(ValidationError):
        Settings()


def test_settings_player_aliases(monkeypatch):
    monkeypatch.setenv(
        "PLEX_PLAYER_ALIASES",
        (
            "{\"machine-1\": [\"Living Room TV\", \"Living Room\"],"
            " \"client-2\": \"Bedroom\"}"
        ),
    )
    settings = Settings()
    assert settings.plex_player_aliases == {
        "machine-1": ["Living Room TV", "Living Room"],
        "client-2": ["Bedroom"],
    }


def test_settings_invalid_aliases(monkeypatch):
    monkeypatch.setenv("PLEX_PLAYER_ALIASES", "not-json")
    with pytest.raises(SettingsError):
        Settings()
