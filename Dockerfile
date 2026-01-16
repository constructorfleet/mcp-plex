# syntax=docker/dockerfile:1.7

# -------------------- Builder --------------------

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder
ENV PATH="/root/.local/bin:$PATH"
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && \
    uv venv "${UV_PROJECT_ENVIRONMENT}" && \
    uv sync --locked --no-install-project --no-editable --link-mode=copy

ADD . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable --link-mode=copy

RUN chmod +x /app/entrypoint.sh

# -------------------- Runtime --------------------

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV UV_PYTHON_INSTALL_DIR=/opt/uv
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

COPY --from=builder --chown=app:app /opt/uv /opt/uv
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /app/mcp_plex ./mcp_plex
COPY --from=builder --chown=app:app /app/entrypoint.sh ./entrypoint.sh
COPY --from=builder /app/pyproject.toml ./pyproject.toml
COPY --from=builder /app/uv.lock ./uv.lock

ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd -g ${APP_GID} app && \
    useradd --no-log-init -r -u ${APP_UID} -g ${APP_GID} -d /home/app app && \
    mkdir -p /home/app /data && \
    chown -R app:app /home/app /data /app /opt/uv

ENV PATH="${UV_PROJECT_ENVIRONMENT}/bin:/usr/local/bin:/usr/bin:/bin"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

USER app

ENTRYPOINT ["./entrypoint.sh"]
CMD ["python", "-m", "mcp_plex.server"]
