<!-- Pytest Coverage Comment:Begin -->
<!-- Pytest Coverage Comment:End -->

# mcp-plex

Plex MCP Tool Server.

## Loader

The `load-data` CLI can load Plex metadata into Qdrant. Run it once with:

```bash
uv run load-data --sample-dir sample-data
```

To keep the loader running continuously, pass `--continuous` and optionally set a
delay between runs:

```bash
uv run load-data --continuous --delay 600
```

## Docker

A Dockerfile is provided to build a GPU-enabled image based on
`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` using [uv](https://github.com/astral-sh/uv)
for dependency management. Build and run the loader:

```bash
docker build -t mcp-plex .
docker run --rm --gpus all mcp-plex --sample-dir /data
```

Use the `--continuous` and `--delay` flags with `docker run` to keep the loader
running in a loop.
