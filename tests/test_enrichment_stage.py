import asyncio
from typing import Any

from mcp_plex.common.types import AggregatedItem, IMDbTitle, TMDBMovie
from mcp_plex.loader.pipeline.channels import (
    IMDbRetryQueue,
    INGEST_DONE,
    MovieBatch,
)
from mcp_plex.loader.pipeline.enrichment import EnrichmentStage


def test_enrichment_stage_logger_name() -> None:
    async def scenario() -> str:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="tmdb",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=25,
            episode_batch_size=10,
        )
        return stage.logger.name

    logger_name = asyncio.run(scenario())
    assert logger_name == "mcp_plex.loader.enrichment"


def test_enrichment_stage_uses_injected_imdb_retry_queue() -> None:
    async def scenario() -> bool:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        retry_queue = IMDbRetryQueue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="tmdb",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=retry_queue,
            movie_batch_size=25,
            episode_batch_size=10,
        )
        return stage.imdb_retry_queue is retry_queue

    same_queue = asyncio.run(scenario())
    assert same_queue is True


def test_enrichment_stage_creates_retry_queue_when_missing() -> None:
    async def scenario() -> IMDbRetryQueue:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="tmdb",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=None,
            movie_batch_size=25,
            episode_batch_size=10,
        )
        return stage.imdb_retry_queue

    retry_queue = asyncio.run(scenario())
    assert isinstance(retry_queue, IMDbRetryQueue)


class _FakeGuid:
    def __init__(self, guid: str) -> None:
        self.id = guid


class _FakeMovie:
    def __init__(
        self,
        rating_key: str,
        *,
        imdb_id: str | None = None,
        tmdb_id: str | None = None,
    ) -> None:
        self.ratingKey = rating_key
        self.guid = f"plex://{rating_key}"
        self.type = "movie"
        self.title = f"Movie {rating_key}"
        self.summary = f"Summary {rating_key}"
        self.year = 2000
        self.addedAt = None
        self.guids = []
        if imdb_id:
            self.guids.append(_FakeGuid(f"imdb://{imdb_id}"))
        if tmdb_id:
            self.guids.append(_FakeGuid(f"tmdb://{tmdb_id}"))
        self.directors: list[Any] = []
        self.writers: list[Any] = []
        self.actors: list[Any] = []
        self.roles: list[Any] = []
        self.genres: list[Any] = []
        self.collections: list[Any] = []


class _FakeClient:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    async def __aenter__(self) -> "_FakeClient":
        self._log.append("enter")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._log.append("exit")


def test_enrichment_stage_enriches_movie_batches_and_emits_chunks(monkeypatch):
    imdb_requests: list[list[str]] = []
    tmdb_requests: list[str] = []
    client_log: list[str] = []

    async def fake_fetch_imdb_batch(client, imdb_ids):
        imdb_requests.append(list(imdb_ids))
        return {
            imdb_id: IMDbTitle(id=imdb_id, type="movie", primaryTitle=f"IMDb {imdb_id}")
            for imdb_id in imdb_ids
        }

    async def fake_fetch_tmdb_movie(client, tmdb_id, api_key):
        tmdb_requests.append(tmdb_id)
        return TMDBMovie.model_validate({
            "id": int(tmdb_id),
            "title": f"TMDb {tmdb_id}",
        })

    monkeypatch.setattr(
        EnrichmentStage, "_handle_episode_batch", lambda self, batch: asyncio.sleep(0)
    )
    monkeypatch.setattr(
        EnrichmentStage, "_handle_sample_batch", lambda self, batch: asyncio.sleep(0)
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_imdb_batch",
        fake_fetch_imdb_batch,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_movie",
        fake_fetch_tmdb_movie,
    )

    async def scenario() -> list[list[AggregatedItem] | None]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()

        stage = EnrichmentStage(
            http_client_factory=lambda: _FakeClient(client_log),
            tmdb_api_key="token",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=2,
            episode_batch_size=10,
        )

        movies = [
            _FakeMovie("1", imdb_id="tt1", tmdb_id="101"),
            _FakeMovie("2", imdb_id="tt2", tmdb_id="102"),
            _FakeMovie("3", imdb_id="tt3", tmdb_id="103"),
        ]
        await ingest_queue.put(MovieBatch(movies=movies))
        await ingest_queue.put(INGEST_DONE)

        await stage.run()

        emitted: list[list[AggregatedItem] | None] = []
        while True:
            payload = await persistence_queue.get()
            emitted.append(payload)
            if payload is None:
                break
        return emitted

    emitted_batches = asyncio.run(scenario())

    assert imdb_requests == [["tt1", "tt2"], ["tt3"]]
    assert tmdb_requests == ["101", "102", "103"]
    assert client_log == ["enter", "exit"]

    assert len(emitted_batches) == 3
    first, second, sentinel = emitted_batches
    assert isinstance(first, list)
    assert isinstance(second, list)
    assert sentinel is None
    assert [item.plex.rating_key for item in first] == ["1", "2"]
    assert [item.plex.rating_key for item in second] == ["3"]
    assert all(item.imdb is not None for item in first + second)
    assert all(item.tmdb is not None for item in first + second)


def test_enrichment_stage_handles_missing_external_ids(monkeypatch):
    imdb_requests: list[list[str]] = []
    tmdb_requests: list[str] = []

    async def fake_fetch_imdb_batch(client, imdb_ids):
        imdb_requests.append(list(imdb_ids))
        return {
            imdb_id: IMDbTitle(id=imdb_id, type="movie", primaryTitle=f"IMDb {imdb_id}")
            for imdb_id in imdb_ids
        }

    async def fake_fetch_tmdb_movie(client, tmdb_id, api_key):
        tmdb_requests.append(tmdb_id)
        return TMDBMovie.model_validate({"id": int(tmdb_id), "title": tmdb_id})

    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_imdb_batch",
        fake_fetch_imdb_batch,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_movie",
        fake_fetch_tmdb_movie,
    )

    async def scenario() -> list[AggregatedItem]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()

        stage = EnrichmentStage(
            http_client_factory=lambda: _FakeClient([]),
            tmdb_api_key="token",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=5,
            episode_batch_size=10,
        )

        movies = [
            _FakeMovie("1", imdb_id="tt1", tmdb_id="201"),
            _FakeMovie("2"),
        ]
        await ingest_queue.put(MovieBatch(movies=movies))
        await ingest_queue.put(INGEST_DONE)

        await stage.run()

        payload = await persistence_queue.get()
        assert isinstance(payload, list)
        await persistence_queue.get()  # sentinel
        return payload

    aggregated = asyncio.run(scenario())

    assert imdb_requests == [["tt1"]]
    assert tmdb_requests == ["201"]
    assert aggregated[0].imdb is not None and aggregated[0].tmdb is not None
    assert aggregated[1].imdb is None and aggregated[1].tmdb is None
