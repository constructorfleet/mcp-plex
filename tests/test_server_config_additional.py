from __future__ import annotations

import pytest

from mcp_plex.server.config import Settings


def test_settings_defaults_reranker_model():
    settings = Settings()
    assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_settings_env_alias_for_reranker_model(monkeypatch):
    monkeypatch.setenv("RERANKER_MODEL", "sentence-transformers/custom")
    settings = Settings()
    assert settings.reranker_model == "sentence-transformers/custom"


def test_parse_aliases_rejects_invalid_json():
    with pytest.raises(ValueError):
        Settings._parse_aliases("not json")


def test_parse_aliases_rejects_unexpected_type():
    with pytest.raises(ValueError):
        Settings._parse_aliases(123)  # type: ignore[arg-type]


def test_parse_aliases_handles_empty_input():
    assert Settings._parse_aliases("") == {}


def test_parse_aliases_rejects_non_collection_json():
    with pytest.raises(ValueError):
        Settings._parse_aliases("123")


def test_parse_aliases_skips_blank_keys():
    assert Settings._parse_aliases({"": ["alias"]}) == {}


def test_items_from_sequence_requires_pairs():
    with pytest.raises(ValueError):
        Settings._items_from_sequence([["only-one"]])


def test_normalize_alias_values_rejects_invalid_type():
    with pytest.raises(ValueError):
        Settings._normalize_alias_values(123)  # type: ignore[arg-type]


def test_items_from_sequence_rejects_invalid_entry_type():
    with pytest.raises(ValueError):
        Settings._items_from_sequence(["string-entry"])  # type: ignore[list-item]
