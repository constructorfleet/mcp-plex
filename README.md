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
- Title search automatically strips leading articles ("The", "A", "An", etc.) and now falls back to hybrid dense/sparse vector similarity so near-miss spellings still resolve (e.g., asking for "The Gentleman" returns "The Gentlemen").
- Recommend similar media from a reference item.
- GPU-enabled data loading and embeddings.

> GPU dependencies are installed automatically on Linux x86_64 hosts. Arm64 (including Apple Silicon dev containers) falls back to the CPU-only embedding stack so setup succeeds everywhere.

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

### IMDb Retry Configuration
The loader exposes CLI flags (and mirrored environment variables) that control
how it retries IMDb lookups and how aggressively it backs off when hitting
`HTTP 429` responses. Use the CLI options during ad-hoc runs or set the
environment variables when deploying via Docker Compose or other orchestrators.

| CLI flag | Environment variable | Default | How it influences retries |
| --- | --- | --- | --- |
| `--imdb-cache` | `IMDB_CACHE` | `imdb_cache.json` | Stores successful IMDb responses on disk so repeated runs can skip network requests. Cached hits never enter the retry pipeline. |
| `--imdb-max-retries` | `IMDB_MAX_RETRIES` | `3` | Number of retry attempts when IMDb returns rate-limit responses. The loader makes `max_retries + 1` total attempts before deferring an ID to the retry queue. |
| `--imdb-backoff` | `IMDB_BACKOFF` | `1.0` | Initial delay in seconds before the next retry. The loader doubles the delay after each failed attempt (1s → 2s → 4s …), slowing down repeated bursts. |
| `--imdb-requests-per-window` | `IMDB_REQUESTS_PER_WINDOW` | disabled | Enables a throttle that only allows the configured number of IMDb requests within the window defined below. Leave unset (`None`) to run without throttling. |
| `--imdb-window-seconds` | `IMDB_WINDOW_SECONDS` | `1.0` | Duration of the sliding window used by the throttle. Combined with the option above it smooths out request spikes when you need to stay under strict quotas. |
| `--imdb-queue` | `IMDB_QUEUE` | `imdb_queue.json` | Path where the retry queue is persisted between runs. IDs that continue to fail after all retries are appended here and reloaded first on the next execution. |

The defaults balance steady throughput with respect for IMDb's limits. For most
home libraries the standard settings work without modification. Persist the
retry queue to a custom path whenever you run multiple loader containers or
want the queue to live on shared storage (for example, mounting `/data/imdb` in
Docker) so unfinished retries survive container rebuilds or host migrations.

#### Aggressive retry policy
```bash
uv run load-data \
  --imdb-max-retries 5 \
  --imdb-backoff 0.5 \
  --imdb-queue /data/imdb/aggressive_queue.json
```
Setting `IMDB_MAX_RETRIES=5` and `IMDB_BACKOFF=0.5` halves the initial delay
and allows two additional retry attempts before falling back to the queue. With
`IMDB_REQUESTS_PER_WINDOW` left unset, the loader does not throttle outbound
requests, so it will recover faster from short spikes at the cost of being more
likely to trigger hard IMDb rate limits.

#### Conservative retry policy
```bash
export IMDB_MAX_RETRIES=2
export IMDB_BACKOFF=2.0
export IMDB_REQUESTS_PER_WINDOW=10
export IMDB_WINDOW_SECONDS=60
export IMDB_QUEUE=/data/imdb/conservative_queue.json
uv run load-data
```
These settings reduce the retry count, double the initial delay, and apply a
token-bucket throttle that only allows 10 requests per minute. They are useful
when sharing an API key across several workers. Persisting the queue to a
shared path makes sure pending IDs continue to retry gradually even if the
container stops.

### Run the MCP Server
Start the FastMCP server over stdio (default):
```bash
uv run mcp-server
```
Expose the server over SSE on port 8000:
```bash
uv run mcp-server --transport sse --bind 0.0.0.0 --port 8000 --mount /mcp
```

Expose the server over streamable HTTP when your MCP client expects a plain
streamed response body instead of SSE events. This mode always requires explicit
bind, port, and mount values provided on the command line or through
environment variables. **Example command:**

```bash
uv run mcp-server --transport streamable-http --bind 0.0.0.0 --port 8800 --mount /mcp
```

Set `MCP_TRANSPORT=streamable-http` along with `MCP_BIND`, `MCP_PORT`, and
`MCP_MOUNT` to configure the same behavior via environment variables. Use SSE
for browser-based connectors or any client that natively supports
Server-Sent Events and wants automatic reconnection. Choose streamable HTTP
for clients that expect a single streaming HTTP response (for example, CLI
tools or proxies that terminate SSE).

Provide `--recommend-user <username>` (or set `PLEX_RECOMMEND_USER`) when the
server should hide items already watched by a specific Plex account from
recommendations. Pair the flag with
`--recommend-history-limit <count>`/`PLEX_RECOMMEND_HISTORY_LIMIT` to constrain
how many watch-history entries the server inspects (defaults to 500) so large
libraries avoid excessive Plex API calls.

The runtime also reads `MCP_TRANSPORT`, `MCP_HOST`, `MCP_PORT`, and `MCP_MOUNT`
environment variables. When set, those values override any conflicting CLI
flags so Docker Compose or other orchestrators can control the exposed MCP
endpoint without editing the container command.

#### Server Configuration

Additional environment variables tune how the server searches and serves
results:

- `CACHE_SIZE` sets the maximum number of distinct query responses cached in
  memory. Increase it when the deployment handles many simultaneous users or
  long-running sessions that repeat complex queries.
- `USE_RERANKER` toggles cross-encoder reranking. Set it to `0` to disable the
  reranker entirely when you want lower latency or do not have GPU capacity.
- `PLEX_PLAYER_ALIASES` provides alternate player names so commands stay
  consistent with custom Plex clients. Supply the value as JSON or as Python
  tuple syntax to align with the server CLI parser. Alias values may reference
  Plex display names as well as machine or client identifiers, and the server
  will resolve the appropriate player in either direction.
- `PLEX_CLIENTS_FILE` points at a YAML, JSON, or XML Plex clients fixture that
  overrides live client discovery. Define the same path with
  `plex_clients_file=/abs/path/clients.yaml` inside a `.env` file or settings
  profile when you prefer configuration files over environment variables. Each
  entry should include the fields that Plex would normally report via the
  `/clients` endpoint so playback control never depends on unstable discovery
  responses.
- `PLEX_RECOMMEND_USER` names a Plex user whose watch history should be
  excluded from similarity recommendations. The server caches that user's
  rating keys and filters them from results so the caller sees only unseen
  titles.
- `PLEX_RECOMMEND_HISTORY_LIMIT` caps how many watch-history records the server
  fetches for the configured user when filtering recommendations. Increase it
  if results still include previously watched items; decrease it to reduce
  Plex API load.

Examples:

```bash
# Disable cross-encoder reranking to prioritize latency.
USE_RERANKER=0 uv run mcp-server

# Expand the in-memory cache to store up to 2,000 results.
CACHE_SIZE=2000 uv run mcp-server

# Map model-friendly player names to Plex devices by display name.
PLEX_PLAYER_ALIASES='{"living_room":"Plex for Roku"}' uv run mcp-server

# Alias keys can also point at Plex machine or client identifiers.
PLEX_PLAYER_ALIASES='{"movie_room":"6B4C9A5E-E333-4DB3-A8E7-49C8F5933EB1"}' \
  uv run mcp-server

# Tuple syntax is also accepted for aliases.
PLEX_PLAYER_ALIASES="[(\"living_room\", \"Plex for Roku\")]" uv run mcp-server

# Load a Plex clients fixture to bypass flaky discovery responses.
PLEX_CLIENTS_FILE=/opt/mcp/config/clients.yaml uv run mcp-server

#### Plex clients fixture

Defining a Plex clients fixture locks the server to a curated set of players so
playback continues even when the `/clients` endpoint returns stale metadata.
The file mirrors the format returned by Plex and may be authored in XML, JSON,
or YAML. Each `<Server>`/object entry maps directly to a `PlexClient` instance
with optional `<Alias>` elements that become friendly names during player
matching. For example:

```xml
<MediaContainer size="2">
  <Server name="Apple TV" address="10.0.12.122" port="32500"
          machineIdentifier="243795C0-C395-4C64-AFD9-E12390C86595"
          product="Plex for Apple TV" protocolCapabilities="playback,playqueues,timeline">
    <Alias>Movie Room TV</Alias>
    <Alias>Movie Room</Alias>
  </Server>
  <Server name="Apple TV" address="10.0.12.94" port="32500"
          machineIdentifier="243795C0-C395-4C64-AFD9-E12390C86212"
          product="Plex for Apple TV" protocolCapabilities="playback,playqueues,timeline">
    <Alias>Office AppleTV</Alias>
    <Alias>Office TV</Alias>
    <Alias>Office</Alias>
  </Server>
</MediaContainer>
```

The same structure works in JSON or YAML using `MediaContainer` and `Server`
keys. When the file is loaded, the server instantiates `PlexClient` objects with
the provided metadata and reuses the alias list when matching playback commands.
```

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
uv run ruff check .
uv run pytest
```

### VS Code Dev Container
The `.devcontainer` setup builds on top of `mcr.microsoft.com/devcontainers/base:ubuntu-22.04`,
copies the `uv` binaries from `ghcr.io/astral-sh/uv`, and runs `uv sync --extra dev` after the
container is created so the local `.venv` matches `uv.lock`. Open the folder in VS Code and choose
**Dev Containers: Reopen in Container** to use it. The definition mounts a persistent `uv-cache`
volume at `/home/vscode/.cache/uv` so Python downloads and wheels survive rebuilds; remove the
volume if you need a fully fresh install. Override the `UV_DISTRO_IMAGE` build argument inside
`.devcontainer/devcontainer.json` to pin a specific uv release (for example,
`ghcr.io/astral-sh/uv:0.9.25`) and run **Dev Containers: Rebuild Container** to apply the change.

## Contributing
Please read [AGENTS.md](AGENTS.md) for commit message conventions and PR
requirements.

## License
Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
