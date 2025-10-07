from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PlexPlayerAliasMap = dict[str, tuple[str, ...]]
RawAliasValue = str | Sequence[Any]
RawAliasItems = list[tuple[Any, RawAliasValue]]
RawAliasMapping = Mapping[str, RawAliasValue]
RawAliases = str | RawAliasMapping | RawAliasItems | None


class Settings(BaseSettings):
    """Application configuration settings."""

    qdrant_url: str | None = Field(default=None, validation_alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    qdrant_host: str | None = Field(default=None, validation_alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, validation_alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(default=6334, validation_alias="QDRANT_GRPC_PORT")
    qdrant_prefer_grpc: bool = Field(
        default=False, validation_alias="QDRANT_PREFER_GRPC"
    )
    qdrant_https: bool | None = Field(default=None, validation_alias="QDRANT_HTTPS")
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
    plex_player_aliases: PlexPlayerAliasMap = Field(
        default_factory=dict, validation_alias="PLEX_PLAYER_ALIASES"
    )

    @field_validator("plex_player_aliases", mode="before")
    @classmethod
    def _parse_aliases(cls, value: RawAliases) -> PlexPlayerAliasMap:
        if value in (None, ""):
            return {}

        if isinstance(value, str):
            try:
                loaded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("PLEX_PLAYER_ALIASES must be valid JSON") from exc
            if not isinstance(loaded, (Mapping, Sequence)):
                raise ValueError(
                    "PLEX_PLAYER_ALIASES JSON must decode to a mapping or sequence"
                )
            value = loaded

        if isinstance(value, Mapping):
            items = list(value.items())
        elif isinstance(value, Sequence):
            items = cls._items_from_sequence(value)
        else:
            raise ValueError("PLEX_PLAYER_ALIASES must be a mapping or sequence")

        parsed: PlexPlayerAliasMap = {}
        for raw_key, raw_aliases in items:
            key = str(raw_key).strip()
            if not key:
                continue
            normalized = cls._normalize_alias_values(raw_aliases)
            if normalized:
                parsed[key] = tuple(normalized)
        return parsed

    @staticmethod
    def _items_from_sequence(value: Sequence[Any]) -> RawAliasItems:
        items: RawAliasItems = []
        for entry in value:
            if isinstance(entry, Mapping):
                items.extend(entry.items())
            elif isinstance(entry, Sequence) and not isinstance(
                entry, (str, bytes, bytearray)
            ):
                entry_list = list(entry)
                if len(entry_list) != 2:
                    raise ValueError(
                        "PLEX_PLAYER_ALIASES sequence entries must contain exactly two items"
                    )
                items.append((entry_list[0], entry_list[1]))
            else:
                raise ValueError(
                    "PLEX_PLAYER_ALIASES sequence entries must be mappings or 2-tuples"
                )
        return items

    @staticmethod
    def _normalize_alias_values(raw_aliases: RawAliasValue) -> list[str]:
        if isinstance(raw_aliases, str):
            values: Sequence[Any] = [raw_aliases]
        elif isinstance(raw_aliases, Sequence) and not isinstance(
            raw_aliases, (str, bytes, bytearray)
        ):
            values = raw_aliases
        else:
            raise ValueError(
                "PLEX_PLAYER_ALIASES values must be strings or iterables of strings"
            )

        normalized: list[str] = []
        for alias in values:
            if alias is None:
                continue
            alias_str = str(alias).strip()
            if alias_str and alias_str not in normalized:
                normalized.append(alias_str)
        return normalized

    model_config = SettingsConfigDict(case_sensitive=False)
