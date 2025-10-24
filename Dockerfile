# syntax=docker/dockerfile:1.7
ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04

FROM ghcr.io/astral-sh/uv:0.4.27 AS uvbin

# -------------------- Builder --------------------
FROM nvidia/cuda:${CUDA_TAG} AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=uvbin /uv /usr/local/bin/uv
COPY --from=uvbin /uvx /usr/local/bin/uvx

# Pin uv’s interpreter install dir and force project env path to /opt/venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:/usr/local/bin:/usr/bin:/bin"

WORKDIR /build

# Prime deps cache
RUN --mount=type=cache,target=/root/.cache/uv true
COPY pyproject.toml uv.lock /build/

# Install CPython 3.12 for uv, create venv at /opt/venv, then sync deps (no project yet)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12 && \
    uv venv "${VIRTUAL_ENV}" && \
    uv pip install --upgrade pip && \
    uv sync \
      --locked \
      --no-editable \
      --no-install-project \
      --link-mode=copy

# Project code
COPY mcp_plex/ /build/mcp_plex/
COPY entrypoint.sh /build/entrypoint.sh

# Install project into the same venv; make sure entrypoint is executable
RUN --mount=type=cache,target=/root/.cache/uv \
    chmod +x /build/entrypoint.sh && \
    uv sync \
      --locked \
      --no-editable \
      --link-mode=copy

# Sanity: verify python is executable and resolved
RUN set -eux; \
    ls -l /opt/venv/bin; \
    readlink -f /opt/venv/bin/python || true; \
    test -x /opt/venv/bin/python

# -------------------- Runtime --------------------
FROM nvidia/cuda:${CUDA_TAG} AS runtime
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# Copy uv-managed CPython + venv + app
COPY --from=builder /opt/uv   /opt/uv
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --from=builder /build/mcp_plex /app/mcp_plex
COPY --from=builder /build/entrypoint.sh /app/entrypoint.sh
# Optional only if read at runtime:
COPY pyproject.toml uv.lock /app/

# Non-root user
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd -g ${APP_GID} app && \
    useradd --no-log-init -r -u ${APP_UID} -g ${APP_GID} -d /home/app app && \
    mkdir -p /home/app /data && \
    chown -R app:app /home/app /data /app /opt/venv /opt/uv && \
    chmod +x /app/entrypoint.sh && \
    # Belt-and-suspenders exec bits
    find /opt/venv/bin -maxdepth 1 -type f -exec chmod 0755 {} \; && \
    find /opt/uv -type f -name "python*" -exec chmod 0755 {} \; || true

ENV VIRTUAL_ENV=/opt/venv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:/usr/local/bin:/usr/bin:/bin"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

USER app

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD python -c "import importlib; importlib.import_module('mcp_plex'); raise SystemExit(0)" || exit 1

ENTRYPOINT ["/usr/bin/tini","-g","--","/app/entrypoint.sh"]