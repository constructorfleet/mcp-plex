"""Compatibility tests for loader public API aliases."""

import importlib

import pytest


def test_loader_pipeline_alias_removed() -> None:
    """Ensure the legacy alias no longer exposes the orchestrator."""

    module = importlib.import_module("mcp_plex.loader")
    module = importlib.reload(module)

    loader_cls = getattr(module, "LoaderPipeline")
    assert hasattr(loader_cls, "execute"), "expected legacy LoaderPipeline class"

    with pytest.raises(AttributeError):
        getattr(module, "LegacyLoaderPipeline")
