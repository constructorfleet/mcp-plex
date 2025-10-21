from pathlib import Path


def test_readme_documents_streamable_http_transport():
    readme = Path("README.md").read_text()

    required_phrases = (
        "--transport streamable-http",
        "--bind",
        "--port",
        "--mount",
        "MCP_BIND",
        "MCP_PORT",
        "MCP_MOUNT",
        "SSE",
        "streamable HTTP",
        "Example",
    )

    assert all(phrase in readme for phrase in required_phrases)
