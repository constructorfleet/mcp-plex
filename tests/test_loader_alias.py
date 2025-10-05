"""Compatibility tests for loader public API aliases."""

import importlib
import warnings


def test_loader_pipeline_alias_emits_deprecation_warning() -> None:
    """The legacy LoaderPipeline export should point at the orchestrator."""

    module = importlib.import_module("mcp_plex.loader")
    module = importlib.reload(module)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        alias = getattr(module, "LoaderPipeline")

    assert alias is module.LoaderOrchestrator
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
