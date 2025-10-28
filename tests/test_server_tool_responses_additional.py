from __future__ import annotations

import asyncio
import inspect
from typing import Any, get_args, get_origin, get_type_hints

import pytest
from pydantic import BaseModel

from mcp_plex import server as server_module
from mcp_plex.server import media as media_helpers


@pytest.mark.parametrize(
    "tool_name",
    [
        "get_media",
        "search_media",
        "query_media",
        "recommend_media_like",
        "recommend_media",
        "new_movies",
        "new_shows",
        "actor_movies",
    ],
)
def test_media_library_tools_expose_pydantic_return_models(tool_name: str) -> None:
    tool = getattr(server_module, tool_name)
    hints = get_type_hints(tool.fn)
    assert "return" in hints, f"{tool_name} is missing a return annotation"
    return_type = hints["return"]

    def _unwrap(tp: Any) -> Any:
        origin = get_origin(tp)
        if origin is None:
            return tp
        if origin in {list, tuple, set}:
            (arg,) = get_args(tp)
            return _unwrap(arg)
        if origin is inspect.Signature.empty:
            return tp
        return tp

    model_type = _unwrap(return_type)
    assert inspect.isclass(model_type), f"{tool_name} return type is not a class"
    assert issubclass(
        model_type, BaseModel
    ), f"{tool_name} should return a Pydantic model"


def test_get_media_returns_pydantic_models_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run() -> None:
        server_module.server.cache.clear()
        sample = {
            "title": "Example Movie",
            "summary": "An illustrative entry for Pydantic conversion.",
            "plex": {"rating_key": "demo"},
        }

        monkeypatch.setattr(
            media_helpers,
            "_get_cached_payload",
            lambda cache, identifier: sample,
        )

        results = await server_module.get_media.fn(identifier="demo")
        assert results, "Expected at least one media item"
        first = results[0]
        assert isinstance(first, BaseModel)
        assert first.model_dump()["title"] == "Example Movie"

    asyncio.run(_run())
