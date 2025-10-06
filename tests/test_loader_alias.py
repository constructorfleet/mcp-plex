"""Compatibility tests for loader public API aliases."""

import importlib

import pytest


def test_loader_pipeline_removed() -> None:
    """Ensure the legacy ``LoaderPipeline`` symbol is no longer exported."""

    module = importlib.import_module("mcp_plex.loader")
    module = importlib.reload(module)

    with pytest.raises(AttributeError):
        getattr(module, "LoaderPipeline")

    with pytest.raises(AttributeError):
        getattr(module, "LegacyLoaderPipeline")
