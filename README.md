![coverage](https://raw.githubusercontent.com/constructorfleet/mcp-plex/coveragebadges/coverage.svg)
![tests](https://raw.githubusercontent.com/constructorfleet/mcp-plex/coveragebadges/tests.svg)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-donate-yellow?logo=buy-me-a-coffee&logoColor=white)](https://www.buymeacoffee.com/constructorfleet)

# mcp-plex

`mcp-plex` is a [Model Context Protocol](https://github.com/modelcontextprotocol) server and data
loader for Plex. It ingests Plex metadata into [Qdrant](https://qdrant.tech/) and exposes
search and recommendation tools that LLM agents can call.

## Features
- Load Plex libraries into a Qdrant vector database.
- Hybrid dense & sparse vector search for media items.
- Recommend similar media from a reference item.
- GPU-enabled data loading and embeddings.

## Installation
1. Install [`uv`](https://github.com/astral-sh/uv).
2. Sync project dependencies (including dev tools) with:
   ```bash
   uv sync --extra dev
   ```

## Usage
### Load Plex Metadata
Load sample data into Qdrant:
```bash
uv run load-data --sample-dir sample-data
```

Run continuously with a delay between runs:
```bash
uv run load-data --continuous --delay 600
```

### Run the MCP Server
Start the FastMCP server to expose Plex tools:
```bash
uv run python -m mcp_plex.server
```

## Docker
A Dockerfile builds a GPU-enabled image based on
`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` using `uv` for dependency
management. Build and run the loader:
```bash
docker build -t mcp-plex .
docker run --rm --gpus all mcp-plex --sample-dir /data
```
Use `--continuous` and `--delay` flags with `docker run` to keep the loader
running in a loop.

## Development
Run linting and tests through `uv`:
```bash
uv run ruff .
uv run pytest
```

## Contributing
Please read [AGENTS.md](AGENTS.md) for commit message conventions and PR
requirements.

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
