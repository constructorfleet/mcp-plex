from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    qdrant_url: str | None = Field(default=None, env="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, env="QDRANT_API_KEY")
    qdrant_host: str | None = Field(default=None, env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, env="QDRANT_GRPC_PORT")
    qdrant_prefer_grpc: bool = Field(default=False, env="QDRANT_PREFER_GRPC")
    qdrant_https: bool | None = Field(default=None, env="QDRANT_HTTPS")
    dense_model: str = Field(
        default="BAAI/bge-small-en-v1.5", env="DENSE_MODEL"
    )
    sparse_model: str = Field(
        default="Qdrant/bm42-all-minilm-l6-v2-attentions", env="SPARSE_MODEL"
    )
    cache_size: int = Field(default=128, env="CACHE_SIZE")
    use_reranker: bool = Field(default=True, env="USE_RERANKER")

    model_config = SettingsConfigDict(case_sensitive=False)
