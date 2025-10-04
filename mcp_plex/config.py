from __future__ import annotations

import json

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    qdrant_url: str | None = Field(
        default=None, validation_alias="QDRANT_URL"
    )
    qdrant_api_key: str | None = Field(
        default=None, validation_alias="QDRANT_API_KEY"
    )
    qdrant_host: str | None = Field(
        default=None, validation_alias="QDRANT_HOST"
    )
    qdrant_port: int = Field(default=6333, validation_alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(
        default=6334, validation_alias="QDRANT_GRPC_PORT"
    )
    qdrant_prefer_grpc: bool = Field(
        default=False, validation_alias="QDRANT_PREFER_GRPC"
    )
    qdrant_https: bool | None = Field(
        default=None, validation_alias="QDRANT_HTTPS"
    )
    dense_model: str = Field(
        default="BAAI/bge-small-en-v1.5", validation_alias="DENSE_MODEL"
    )
    sparse_model: str = Field(
        default="Qdrant/bm42-all-minilm-l6-v2-attentions",
        validation_alias="SPARSE_MODEL",
    )
    cache_size: int = Field(default=128, validation_alias="CACHE_SIZE")
    use_reranker: bool = Field(default=True, validation_alias="USE_RERANKER")
    plex_url: AnyHttpUrl | None = Field(default=None, validation_alias="PLEX_URL")
    plex_token: str | None = Field(default=None, validation_alias="PLEX_TOKEN")
    plex_player_aliases: dict[str, str] = Field(
        default_factory=dict, validation_alias="PLEX_PLAYER_ALIASES"
    )

    @field_validator("plex_player_aliases", mode="before")
    @classmethod
    def _parse_aliases(cls, value: object) -> dict[str, str]:
        if value in (None, ""):
            return {}
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("PLEX_PLAYER_ALIASES must be valid JSON") from exc
        if isinstance(value, dict):
            return {str(k): str(v) for k, v in value.items()}
        raise TypeError("PLEX_PLAYER_ALIASES must be a mapping or JSON object")

    model_config = SettingsConfigDict(case_sensitive=False)
