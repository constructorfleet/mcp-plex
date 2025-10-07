"""Command-line interface for the loader pipeline."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click

from . import DEFAULT_QDRANT_UPSERT_BUFFER_SIZE, load_media


@click.command()
@click.option(
    "--plex-url",
    envvar="PLEX_URL",
    show_envvar=True,
    required=True,
    help="Plex base URL",
)
@click.option(
    "--plex-token",
    envvar="PLEX_TOKEN",
    show_envvar=True,
    required=True,
    help="Plex API token",
)
@click.option(
    "--tmdb-api-key",
    envvar="TMDB_API_KEY",
    show_envvar=True,
    required=True,
    help="TMDb API key",
)
@click.option(
    "--sample-dir",
    type=click.Path(path_type=Path),
    required=False,
    help="Directory containing sample data instead of live Plex access",
)
@click.option(
    "--qdrant-url",
    envvar="QDRANT_URL",
    show_envvar=True,
    required=False,
    help="Qdrant URL or path",
)
@click.option(
    "--qdrant-api-key",
    envvar="QDRANT_API_KEY",
    show_envvar=True,
    required=False,
    help="Qdrant API key",
)
@click.option(
    "--qdrant-host",
    envvar="QDRANT_HOST",
    show_envvar=True,
    required=False,
    help="Qdrant host",
)
@click.option(
    "--qdrant-port",
    envvar="QDRANT_PORT",
    show_envvar=True,
    type=int,
    default=6333,
    show_default=True,
    required=False,
    help="Qdrant HTTP port",
)
@click.option(
    "--qdrant-grpc-port",
    envvar="QDRANT_GRPC_PORT",
    show_envvar=True,
    type=int,
    default=6334,
    show_default=True,
    required=False,
    help="Qdrant gRPC port",
)
@click.option(
    "--qdrant-https/--no-qdrant-https",
    envvar="QDRANT_HTTPS",
    show_envvar=True,
    default=False,
    show_default=True,
    help="Use HTTPS when connecting to Qdrant",
)
@click.option(
    "--qdrant-prefer-grpc/--no-qdrant-prefer-grpc",
    envvar="QDRANT_PREFER_GRPC",
    show_envvar=True,
    default=False,
    show_default=True,
    help="Prefer gRPC for Qdrant operations when available",
)
@click.option(
    "--upsert-buffer-size",
    envvar="QDRANT_UPSERT_BUFFER_SIZE",
    show_envvar=True,
    type=int,
    default=DEFAULT_QDRANT_UPSERT_BUFFER_SIZE,
    show_default=True,
    help="Number of points to buffer before dispatching upsert tasks",
)
@click.option(
    "--plex-chunk-size",
    envvar="PLEX_CHUNK_SIZE",
    show_envvar=True,
    type=int,
    default=20,
    show_default=True,
    help="Number of Plex items to fetch per API call",
)
@click.option(
    "--enrichment-batch-size",
    envvar="ENRICHMENT_BATCH_SIZE",
    show_envvar=True,
    type=int,
    default=5,
    show_default=True,
    help="Number of items to enrich concurrently",
)
@click.option(
    "--enrichment-workers",
    envvar="ENRICHMENT_WORKERS",
    show_envvar=True,
    type=int,
    default=5,
    show_default=True,
    help="Number of enrichment workers",
)
@click.option(
    "--dense-model",
    envvar="DENSE_MODEL",
    show_envvar=True,
    default="BAAI/bge-small-en-v1.5",
    show_default=True,
    help="Dense embedding model name",
)
@click.option(
    "--sparse-model",
    envvar="SPARSE_MODEL",
    show_envvar=True,
    default="Qdrant/bm42-all-minilm-l6-v2-attentions",
    show_default=True,
    help="Sparse embedding model name",
)
@click.option(
    "--continuous",
    is_flag=True,
    help="Continuously run the loader",
    show_default=True,
    default=False,
    required=False,
)
@click.option(
    "--log-level",
    envvar="LOG_LEVEL",
    show_envvar=True,
    type=click.Choice(
        ["critical", "error", "warning", "info", "debug", "notset"],
        case_sensitive=False,
    ),
    default="info",
    show_default=True,
    help="Logging level for console output",
)
@click.option(
    "--delay",
    type=click.FloatRange(min=0.0),
    default=300.0,
    show_default=True,
    required=False,
    help="Delay between runs in seconds when using --continuous",
)
@click.option(
    "--imdb-cache",
    envvar="IMDB_CACHE",
    show_envvar=True,
    type=click.Path(path_type=Path),
    default=Path("imdb_cache.json"),
    show_default=True,
    help="Path to persistent IMDb response cache",
)
@click.option(
    "--imdb-max-retries",
    envvar="IMDB_MAX_RETRIES",
    show_envvar=True,
    type=int,
    default=3,
    show_default=True,
    help="Maximum retries for IMDb requests returning HTTP 429",
)
@click.option(
    "--imdb-backoff",
    envvar="IMDB_BACKOFF",
    show_envvar=True,
    type=float,
    default=1.0,
    show_default=True,
    help="Initial backoff delay in seconds for IMDb retries",
)
@click.option(
    "--imdb-requests-per-window",
    envvar="IMDB_REQUESTS_PER_WINDOW",
    show_envvar=True,
    type=int,
    default=None,
    help="Maximum IMDb requests per rate-limit window (set to disable)",
)
@click.option(
    "--imdb-window-seconds",
    envvar="IMDB_WINDOW_SECONDS",
    show_envvar=True,
    type=float,
    default=1.0,
    show_default=True,
    help="Duration in seconds for the IMDb rate-limit window",
)
@click.option(
    "--imdb-queue",
    envvar="IMDB_QUEUE",
    show_envvar=True,
    type=click.Path(path_type=Path),
    default=Path("imdb_queue.json"),
    show_default=True,
    help="Path to persistent IMDb retry queue",
)
def main(
    plex_url: str | None,
    plex_token: str | None,
    tmdb_api_key: str | None,
    sample_dir: Path | None,
    qdrant_url: str | None,
    qdrant_api_key: str | None,
    qdrant_host: str | None,
    qdrant_port: int,
    qdrant_grpc_port: int,
    qdrant_https: bool,
    qdrant_prefer_grpc: bool,
    upsert_buffer_size: int,
    plex_chunk_size: int,
    enrichment_batch_size: int,
    enrichment_workers: int,
    dense_model: str,
    sparse_model: str,
    continuous: bool,
    delay: float,
    imdb_cache: Path,
    imdb_max_retries: int,
    imdb_backoff: float,
    imdb_requests_per_window: int | None,
    imdb_window_seconds: float,
    imdb_queue: Path,
    log_level: str,
) -> None:
    """Entry-point for the ``load-data`` script."""

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    asyncio.run(
        load_media(
            plex_url=plex_url,
            plex_token=plex_token,
            tmdb_api_key=tmdb_api_key,
            sample_dir=sample_dir,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_grpc_port=qdrant_grpc_port,
            qdrant_https=qdrant_https,
            qdrant_prefer_grpc=qdrant_prefer_grpc,
            dense_model_name=dense_model,
            sparse_model_name=sparse_model,
            continuous=continuous,
            delay=delay,
            imdb_cache=imdb_cache,
            imdb_max_retries=imdb_max_retries,
            imdb_backoff=imdb_backoff,
            imdb_requests_per_window=imdb_requests_per_window,
            imdb_window_seconds=imdb_window_seconds,
            imdb_queue=imdb_queue,
            upsert_buffer_size=upsert_buffer_size,
            plex_chunk_size=plex_chunk_size,
            enrichment_batch_size=enrichment_batch_size,
            enrichment_workers=enrichment_workers,
        )
    )


if __name__ == "__main__":
    main()
