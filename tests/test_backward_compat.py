from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name, expected_module_name, attribute_names",
    [
        ("mcp_plex.cache", "mcp_plex.common.cache", ["MediaCache"]),
        ("mcp_plex.imdb_cache", "mcp_plex.loader.imdb_cache", ["IMDbCache"]),
        ("mcp_plex.config", "mcp_plex.server.config", ["Settings"]),
    ],
)
def test_legacy_module_shims(
    module_name: str, expected_module_name: str, attribute_names: list[str]
) -> None:
    with pytest.deprecated_call():
        legacy_module = importlib.import_module(module_name)

    expected_module = importlib.import_module(expected_module_name)

    for name in attribute_names:
        assert getattr(legacy_module, name) is getattr(expected_module, name)


def test_legacy_types_module_exports() -> None:
    with pytest.deprecated_call():
        legacy_types = importlib.import_module("mcp_plex.types")

    expected_types = importlib.import_module("mcp_plex.common.types")

    assert legacy_types.__all__ == expected_types.__all__

    for name in expected_types.__all__:
        assert getattr(legacy_types, name) is getattr(expected_types, name)
