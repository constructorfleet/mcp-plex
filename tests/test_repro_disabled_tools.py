from __future__ import annotations

from mcp_plex.server.config import Settings

def test_disabled_tools_comma_separated(monkeypatch):
    monkeypatch.setenv("DISABLED_TOOLS", "tool1,tool2, tool3")
    # This currently might fail or produce a list with one string ["tool1,tool2, tool3"]
    # if Pydantic doesn't split it.
    settings = Settings()
    assert settings.disabled_tools == ["tool1", "tool2", "tool3"]

if __name__ == "__main__":
    pytest.main([__file__])
