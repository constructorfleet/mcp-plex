# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder
# syntax=docker/dockerfile:1.7
ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04

ENV PATH="/root/.local/bin:$PATH"
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

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
WORKDIR /build