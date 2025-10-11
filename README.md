![coverage](https://raw.githubusercontent.com/constructorfleet/mcp-plex/coveragebadges/coverage.svg)
![tests](https://raw.githubusercontent.com/constructorfleet/mcp-plex/coveragebadges/tests.svg)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-donate-yellow?logo=buy-me-a-coffee&logoColor=white)](https://www.buymeacoffee.com/constructorfleet)

# mcp-plex

`mcp-plex` turns your Plex library into a searchable vector database that LLM agents can query.
It ingests Plex metadata into [Qdrant](https://qdrant.tech/) and exposes search and recommendation
tools through the [Model Context Protocol](https://github.com/modelcontextprotocol).

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

### IMDb Retry Queue
When IMDb lookups continue to return HTTP 429 after the configured retries,
their IDs are added to a small queue (`imdb_queue.json` by default). The queue
is persisted after each run and reloaded on the next run so pending IDs are
retried before normal processing.

### Run the MCP Server
Start the FastMCP server over stdio (default):
```bash
uv run mcp-server
```
Expose the server over SSE on port 8000:
```bash
uv run mcp-server --transport sse --bind 0.0.0.0 --port 8000 --mount /mcp
```

The runtime also reads `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, and `MCP_MOUNT`
environment variables. When set, those values override any conflicting CLI
flags so Docker Compose or other orchestrators can control the exposed MCP
endpoint without editing the container command.

### Embedding Models

Both the loader and server default to `BAAI/bge-small-en-v1.5` for dense
embeddings and `Qdrant/bm42-all-minilm-l6-v2-attentions` for sparse embeddings.
Override these by setting `DENSE_MODEL`/`SPARSE_MODEL` environment variables or
using `--dense-model`/`--sparse-model` CLI options:

```bash
uv run load-data --dense-model my-dense --sparse-model my-sparse
uv run mcp-server --dense-model my-dense --sparse-model my-sparse
```

Cross-encoder reranking defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Set `RERANKER_MODEL` or pass `--reranker-model` to point at a different model:

```bash
uv run mcp-server --reranker-model sentence-transformers/ms-marco-mini
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

### Dependency Layer Workflow

Docker builds now install dependencies using `docker/pyproject.deps.toml`, a
manifest that mirrors the application's runtime requirements. Keep the
`[project]` metadata—including the `version`—identical to the root
`pyproject.toml` so `uv sync` can reuse the shared `uv.lock` without
validation errors. The `Dockerfile` copies that manifest and `uv.lock` first,
runs `uv sync --no-dev --frozen --manifest-path pyproject.deps.toml`, and only
then copies the rest of the source tree. This keeps the heavy dependency layer
cached even when the application version changes.

When cutting a release:

1. Update dependencies as needed and refresh `uv.lock` with `uv lock`.
2. Build the image twice to confirm the dependency layer cache hit:
   ```bash
   docker build -t mcp-plex:release-test-1 .
   docker build -t mcp-plex:release-test-2 .
   ```
   The second build should reuse the `uv sync` layer as long as
   `docker/pyproject.deps.toml` and `uv.lock` are unchanged.
3. Tag and push the final release image once satisfied with the cache behavior.

## Docker Compose
The included `docker-compose.yml` launches both Qdrant and the MCP server.

1. Set `PLEX_URL`, `PLEX_TOKEN`, and `TMDB_API_KEY` in your environment (or a `.env` file).
2. Start the services:
   ```bash
   docker compose up --build
   ```
3. (Optional) Load sample data into Qdrant:
   ```bash
   docker compose run --rm loader load-data --sample-dir sample-data
   ```

The server will connect to the `qdrant` service at `http://qdrant:6333` and
expose an SSE endpoint at `http://localhost:8000/mcp`.

### Qdrant Configuration

Connection settings can be provided via environment variables:

- `QDRANT_URL` – full URL or SQLite path.
- `QDRANT_HOST`/`QDRANT_PORT` – HTTP host and port.
- `QDRANT_GRPC_PORT` – gRPC port.
- `QDRANT_HTTPS` – set to `1` to enable HTTPS.
- `QDRANT_PREFER_GRPC` – set to `1` to prefer gRPC.

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
