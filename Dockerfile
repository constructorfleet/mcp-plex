# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ENV PATH="/root/.local/bin:$PATH"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable --link-mode=copy

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --link-mode=copy

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV PATH="/root/.local/bin:$PATH"
ENV UV_PYTHON_INSTALL_DIR=/opt/uv

ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app

RUN useradd --system --create-home app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /opt/uv /opt/uv
COPY --from=builder --chown=app:app /app/mcp_plex ./mcp_plex
COPY --from=builder --chown=app:app /app/entrypoint.sh ./entrypoint.sh
COPY --from=builder /app/pyproject.toml ./pyproject.toml
COPY --from=builder --chown=app:app /app/uv.lock ./uv.lock

USER app

ENTRYPOINT ["./entrypoint.sh"]
CMD ["load-data"]
