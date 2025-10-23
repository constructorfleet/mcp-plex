# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS base

ENV UV_LINK_MODE=copy
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

FROM base AS builder
ENV PATH="/root/.local/bin:$PATH"
COPY docker/pyproject.deps.toml ./pyproject.toml
COPY uv.lock ./uv.lock
RUN uv sync --no-dev --frozen && mv pyproject.toml pyproject.deps.toml

COPY mcp_plex/ ./mcp_plex/
COPY entrypoint.sh ./entrypoint.sh

FROM base AS runtime
WORKDIR /app
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/mcp_plex ./mcp_plex
COPY --from=builder /app/entrypoint.sh ./entrypoint.sh
COPY pyproject.toml uv.lock ./

ENTRYPOINT ["./entrypoint.sh"]
CMD ["load-data"]
