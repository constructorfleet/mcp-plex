# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY docker/pyproject.deps.toml ./pyproject.toml
COPY uv.lock ./
RUN uv sync --no-dev --frozen && mv pyproject.toml pyproject.deps.toml

COPY . .
ENTRYPOINT ["./entrypoint.sh"]
CMD ["load-data"]
