import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsError

from mcp_plex.server.config import Settings


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
        ('{"machine-1": ["Living Room TV", "Living Room"], "client-2": "Bedroom"}'),
    )
    settings = Settings()
    assert settings.plex_player_aliases == {
        "machine-1": ("Living Room TV", "Living Room"),
        "client-2": ("Bedroom",),
    }


def test_settings_invalid_aliases(monkeypatch):
    monkeypatch.setenv("PLEX_PLAYER_ALIASES", "not-json")
    with pytest.raises(SettingsError):
        Settings()


def test_settings_aliases_from_mapping():
    settings = Settings.model_validate(
        {
            "PLEX_PLAYER_ALIASES": {
                "machine-1": [" Living Room ", "Living Room"],
                "client-2": "Bedroom",
            }
        }
    )
    assert settings.plex_player_aliases == {
        "machine-1": ("Living Room",),
        "client-2": ("Bedroom",),
    }


def test_settings_aliases_from_sequence():
    settings = Settings.model_validate(
        {
            "PLEX_PLAYER_ALIASES": [
                ("machine-1", ("Living Room", "Living Room TV")),
                {"client-2": ["Bedroom", None]},
            ]
        }
    )
    assert settings.plex_player_aliases == {
        "machine-1": ("Living Room", "Living Room TV"),
        "client-2": ("Bedroom",),
    }


def test_settings_invalid_alias_sequence():
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {"PLEX_PLAYER_ALIASES": [("machine-1", "Living Room", "Extra")]}
        )


def test_settings_disabled_tools_env(monkeypatch):
    monkeypatch.setenv("DISABLED_TOOLS", "tool1, tool2")
    settings = Settings()
    assert settings.disabled_tools == ["tool1", "tool2"]


def test_settings_disabled_tools_json_env(monkeypatch):
    monkeypatch.setenv("DISABLED_TOOLS", '["tool3", "tool4"]')
    settings = Settings()
    assert settings.disabled_tools == ["tool3", "tool4"]


def test_settings_disabled_tools_empty_env(monkeypatch):
    monkeypatch.setenv("DISABLED_TOOLS", "")
    settings = Settings()
    assert settings.disabled_tools == []
