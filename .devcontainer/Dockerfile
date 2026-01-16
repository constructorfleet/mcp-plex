# syntax=docker/dockerfile:1.7
ARG UV_DISTRO_IMAGE=ghcr.io/astral-sh/uv:latest

FROM ${UV_DISTRO_IMAGE} AS uv

FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
COPY --from=uv /uvx /usr/local/bin/uvx

ENV UV_TOOL_BIN_DIR=/usr/local/bin \
    UV_LINK_MODE=copy

WORKDIR /workspaces/mcp-plex