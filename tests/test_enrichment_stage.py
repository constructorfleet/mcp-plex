import asyncio
import logging
from typing import Any

import pytest

from mcp_plex.common.types import (
    AggregatedItem,
    IMDbTitle,
    PlexItem,
    TMDBEpisode,
    TMDBMovie,
    TMDBShow,
)
from mcp_plex.loader.pipeline.channels import (
    EpisodeBatch,
    IMDbRetryQueue,
    INGEST_DONE,
    PERSIST_DONE,
    MovieBatch,
    SampleBatch,
)
from mcp_plex.loader.pipeline.enrichment import (
    EnrichmentStage,
    _RequestThrottler,
    _fetch_imdb_batch,
)


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


def test_enrich_movies_runs_tmdb_and_imdb_requests_in_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: dict[str, asyncio.Event] = {}

    async def fake_fetch_imdb_batch(client, imdb_ids, **kwargs):
        events["imdb_started"].set()
        await asyncio.wait_for(events["tmdb_started"].wait(), timeout=1)
        return {
            imdb_id: IMDbTitle(
                id=imdb_id,
                type="movie",
                primaryTitle=f"IMDb {imdb_id}",
            )
            for imdb_id in imdb_ids
        }

    async def fake_fetch_tmdb_movie(client, tmdb_id, api_key):
        events["tmdb_started"].set()
        await asyncio.wait_for(events["imdb_started"].wait(), timeout=1)
        return TMDBMovie.model_validate({
            "id": int(tmdb_id),
            "title": f"TMDb {tmdb_id}",
        })

    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_imdb_batch",
        fake_fetch_imdb_batch,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_movie",
        fake_fetch_tmdb_movie,
    )

    async def scenario() -> list[AggregatedItem]:
        events["imdb_started"] = asyncio.Event()
        events["tmdb_started"] = asyncio.Event()
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="token",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=5,
            episode_batch_size=5,
        )
        movie = _FakeMovie("1", imdb_id="tt0001", tmdb_id="101")
        return await asyncio.wait_for(
            stage._enrich_movies(object(), [movie]), timeout=1
        )

    result = asyncio.run(scenario())

    assert result[0].imdb is not None
    assert result[0].tmdb is not None


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


class _FakeShow:
    def __init__(self, rating_key: str, *, tmdb_id: str | None = None) -> None:
        self.ratingKey = rating_key
        self.guid = f"plex://show/{rating_key}"
        self.type = "show"
        self.title = f"Show {rating_key}"
        self.summary = f"Show summary {rating_key}"
        self.year = 2010
        self.addedAt = None
        self.guids = []
        if tmdb_id:
            self.guids.append(_FakeGuid(f"tmdb://{tmdb_id}"))
        self.genres: list[Any] = []
        self.collections: list[Any] = []


class _FakeEpisode:
    def __init__(
        self,
        rating_key: str,
        *,
        show: _FakeShow,
        season_index: int,
        episode_index: int,
        imdb_id: str | None = None,
    ) -> None:
        self.ratingKey = rating_key
        self.guid = f"plex://episode/{rating_key}"
        self.type = "episode"
        self.title = f"Episode {rating_key}"
        self.summary = f"Episode summary {rating_key}"
        self.year = 2020
        self.parentIndex = season_index
        self.parentTitle = f"Season {season_index}"
        self.parentYear = 2020
        self.index = episode_index
        self.addedAt = None
        self.grandparentTitle = show.title
        self.guids = []
        if imdb_id:
            self.guids.append(_FakeGuid(f"imdb://{imdb_id}"))
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

    async def fake_fetch_imdb_batch(client, imdb_ids, **kwargs):
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
            if payload in (None, PERSIST_DONE):
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
    assert sentinel is PERSIST_DONE
    assert [item.plex.rating_key for item in first] == ["1", "2"]
    assert [item.plex.rating_key for item in second] == ["3"]
    assert all(item.imdb is not None for item in first + second)
    assert all(item.tmdb is not None for item in first + second)


def test_enrichment_stage_handles_missing_external_ids(monkeypatch):
    imdb_requests: list[list[str]] = []
    tmdb_requests: list[str] = []

    async def fake_fetch_imdb_batch(client, imdb_ids, **kwargs):
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


def test_enrichment_stage_skips_tmdb_fetch_without_api_key(monkeypatch):
    movie_calls: list[str] = []
    show_calls: list[str] = []
    episode_calls: list[tuple[int, int, int]] = []

    async def fake_fetch_imdb_batch(client, imdb_ids, **kwargs):
        return {
            imdb_id: IMDbTitle(
                id=imdb_id,
                type="movie" if imdb_id.startswith("ttm") else "tvEpisode",
                primaryTitle=f"IMDb {imdb_id}",
            )
            for imdb_id in imdb_ids
        }

    async def fake_fetch_tmdb_movie(client, tmdb_id, api_key):
        movie_calls.append(tmdb_id)
        return TMDBMovie.model_validate({"id": int(tmdb_id), "title": tmdb_id})

    async def fake_fetch_tmdb_show(client, tmdb_id, api_key):
        show_calls.append(tmdb_id)
        return TMDBShow.model_validate({"id": int(tmdb_id), "name": tmdb_id, "seasons": []})

    async def fake_fetch_tmdb_episode(client, show_id, season_number, episode_number, api_key):
        episode_calls.append((show_id, season_number, episode_number))
        return TMDBEpisode.model_validate(
            {
                "id": show_id * 1000 + season_number * 100 + episode_number,
                "name": f"Episode {episode_number}",
                "season_number": season_number,
                "episode_number": episode_number,
            }
        )

    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_imdb_batch",
        fake_fetch_imdb_batch,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_movie",
        fake_fetch_tmdb_movie,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_show",
        fake_fetch_tmdb_show,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_episode",
        fake_fetch_tmdb_episode,
    )

    async def scenario() -> list[list[AggregatedItem] | None]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()

        stage = EnrichmentStage(
            http_client_factory=lambda: _FakeClient([]),
            tmdb_api_key=None,
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=3,
            episode_batch_size=3,
        )

        movies = [
            _FakeMovie("m1", imdb_id="ttm1", tmdb_id="401"),
            _FakeMovie("m2", imdb_id="ttm2", tmdb_id="402"),
        ]
        show = _FakeShow("show", tmdb_id="501")
        episodes = [
            _FakeEpisode(
                "e1",
                show=show,
                season_index=1,
                episode_index=1,
                imdb_id="tte1",
            ),
            _FakeEpisode(
                "e2",
                show=show,
                season_index=1,
                episode_index=2,
                imdb_id="tte2",
            ),
        ]

        await ingest_queue.put(MovieBatch(movies=movies))
        await ingest_queue.put(EpisodeBatch(show=show, episodes=episodes))
        await ingest_queue.put(INGEST_DONE)

        await stage.run()

        payloads: list[list[AggregatedItem] | None] = []
        while True:
            payload = await persistence_queue.get()
            payloads.append(payload)
            if payload is PERSIST_DONE:
                break
        return payloads

    payloads = asyncio.run(scenario())

    assert movie_calls == []
    assert show_calls == []
    assert episode_calls == []

    assert len(payloads) == 3
    movie_payload, episode_payload, sentinel = payloads
    assert isinstance(movie_payload, list)
    assert isinstance(episode_payload, list)
    assert sentinel is PERSIST_DONE

    assert all(item.tmdb is None for item in movie_payload)
    assert all(item.tmdb is None for item in episode_payload)
    assert {item.imdb.id for item in movie_payload if item.imdb} == {"ttm1", "ttm2"}
    assert {item.imdb.id for item in episode_payload if item.imdb} == {"tte1", "tte2"}


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - tiny shim
        self.messages.append(record.getMessage())


class _RecordingQueue(asyncio.Queue):
    def __init__(self) -> None:
        super().__init__()
        self.put_payloads: list[Any] = []

    async def put(self, item: Any) -> None:  # type: ignore[override]
        self.put_payloads.append(item)
        await super().put(item)

    def put_nowait(self, item: Any) -> None:  # type: ignore[override]
        self.put_payloads.append(item)
        super().put_nowait(item)


def test_enrichment_stage_caches_tmdb_show_results(monkeypatch):
    show_requests: list[tuple[str, str]] = []
    episode_chunk_requests: list[tuple[int, tuple[str, ...], str]] = []
    fallback_requests: list[tuple[int, int, int, str]] = []

    async def fake_fetch_tmdb_show(client, tmdb_id, api_key):
        show_requests.append((tmdb_id, api_key))
        return TMDBShow.model_validate(
            {
                "id": int(tmdb_id),
                "name": f"Show {tmdb_id}",
                "seasons": [
                    {"season_number": 1, "name": "Season 1", "air_date": "2020-01-01"}
                ],
            }
        )

    async def fake_fetch_tmdb_episode_chunk(client, show_id, append_paths, api_key):
        episode_chunk_requests.append((show_id, tuple(append_paths), api_key))
        results: dict[str, TMDBEpisode] = {}
        for path in append_paths:
            parts = path.split("/")
            season = int(parts[1])
            episode = int(parts[3])
            results[path] = TMDBEpisode.model_validate(
                {
                    "id": show_id * 1000 + season * 100 + episode,
                    "name": f"Episode {episode}",
                    "season_number": season,
                    "episode_number": episode,
                }
            )
        return results

    async def fake_fetch_tmdb_episode(client, show_id, season, episode, api_key):
        fallback_requests.append((show_id, season, episode, api_key))
        return TMDBEpisode.model_validate(
            {
                "id": show_id * 1000 + season * 100 + episode,
                "name": f"Episode {episode}",
                "season_number": season,
                "episode_number": episode,
            }
        )

    async def fake_fetch_imdb_batch(client, imdb_ids, **kwargs):
        return {
            imdb_id: IMDbTitle(
                id=imdb_id,
                type="tvEpisode",
                primaryTitle=f"IMDb {imdb_id}",
            )
            for imdb_id in imdb_ids
        }

    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_show",
        fake_fetch_tmdb_show,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_episode_chunk",
        fake_fetch_tmdb_episode_chunk,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_episode",
        fake_fetch_tmdb_episode,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_imdb_batch",
        fake_fetch_imdb_batch,
    )

    async def scenario() -> list[list[AggregatedItem] | None]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="token",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=3,
            episode_batch_size=2,
        )

        show = _FakeShow("show", tmdb_id="301")
        episodes_first = [
            _FakeEpisode("e1", show=show, season_index=1, episode_index=1, imdb_id="ttA"),
            _FakeEpisode("e2", show=show, season_index=1, episode_index=2, imdb_id="ttB"),
        ]
        episodes_second = [
            _FakeEpisode("e3", show=show, season_index=1, episode_index=3, imdb_id="ttC"),
        ]

        await ingest_queue.put(EpisodeBatch(show=show, episodes=episodes_first))
        await ingest_queue.put(EpisodeBatch(show=show, episodes=episodes_second))
        await ingest_queue.put(INGEST_DONE)

        await stage.run()

        payloads: list[list[AggregatedItem] | None] = []
        while True:
            payload = await persistence_queue.get()
            payloads.append(payload)
            if payload in (None, PERSIST_DONE):
                break
        return payloads

    payloads = asyncio.run(scenario())

    assert show_requests == [("301", "token")]
    assert episode_chunk_requests == [
        (301, ("season/1/episode/1", "season/1/episode/2"), "token"),
        (301, ("season/1/episode/3",), "token"),
    ]
    assert fallback_requests == []

    assert len(payloads) == 3
    first, second, sentinel = payloads
    assert sentinel is PERSIST_DONE
    assert [item.plex.rating_key for item in first] == ["e1", "e2"]
    assert [item.plex.rating_key for item in second] == ["e3"]
    assert all(item.tmdb for item in first + second)
    assert [item.tmdb.episode_number for item in first + second if isinstance(item.tmdb, TMDBEpisode)] == [1, 2, 3]


def test_enrichment_stage_falls_back_to_individual_episode_fetch(monkeypatch):
    show_requests: list[str] = []
    chunk_requests: list[tuple[int, tuple[str, ...], str]] = []
    fallback_requests: list[tuple[int, int, int, str]] = []

    async def fake_fetch_tmdb_show(client, tmdb_id, api_key):
        show_requests.append(tmdb_id)
        return TMDBShow.model_validate(
            {
                "id": int(tmdb_id),
                "name": f"Show {tmdb_id}",
                "seasons": [
                    {"season_number": 1, "name": "Season 1", "air_date": "2020-01-01"}
                ],
            }
        )

    async def fake_fetch_tmdb_episode_chunk(client, show_id, append_paths, api_key):
        chunk_requests.append((show_id, tuple(append_paths), api_key))
        return {}

    async def fake_fetch_tmdb_episode(client, show_id, season, episode, api_key):
        fallback_requests.append((show_id, season, episode, api_key))
        return TMDBEpisode.model_validate(
            {
                "id": show_id * 1000 + season * 100 + episode,
                "name": f"Episode {episode}",
                "season_number": season,
                "episode_number": episode,
            }
        )

    async def fake_fetch_imdb_batch(client, imdb_ids, **kwargs):
        return {
            imdb_id: IMDbTitle(
                id=imdb_id,
                type="tvEpisode",
                primaryTitle=f"IMDb {imdb_id}",
            )
            for imdb_id in imdb_ids
        }

    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_show",
        fake_fetch_tmdb_show,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_episode_chunk",
        fake_fetch_tmdb_episode_chunk,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_tmdb_episode",
        fake_fetch_tmdb_episode,
    )
    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_imdb_batch",
        fake_fetch_imdb_batch,
    )

    async def scenario() -> list[list[AggregatedItem] | None]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="token",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=3,
            episode_batch_size=3,
        )

        show = _FakeShow("show", tmdb_id="302")
        episodes = [
            _FakeEpisode("e1", show=show, season_index=1, episode_index=1, imdb_id="ttA"),
            _FakeEpisode("e2", show=show, season_index=1, episode_index=2, imdb_id="ttB"),
        ]

        await ingest_queue.put(EpisodeBatch(show=show, episodes=episodes))
        await ingest_queue.put(INGEST_DONE)

        await stage.run()

        payloads: list[list[AggregatedItem] | None] = []
        while True:
            payload = await persistence_queue.get()
            payloads.append(payload)
            if payload in (None, PERSIST_DONE):
                break
        return payloads

    payloads = asyncio.run(scenario())

    assert show_requests == ["302"]
    assert chunk_requests == [(302, ("season/1/episode/1", "season/1/episode/2"), "token")]
    assert fallback_requests == [
        (302, 1, 1, "token"),
        (302, 1, 2, "token"),
    ]

    assert len(payloads) == 2
    first, sentinel = payloads
    assert sentinel is PERSIST_DONE
    assert [
        item.tmdb.episode_number
        for item in first
        if isinstance(item.tmdb, TMDBEpisode)
    ] == [1, 2]

def test_enrichment_stage_sample_batches_pass_through(monkeypatch):
    handler = _ListHandler()

    async def scenario() -> tuple[list[list[AggregatedItem] | None], list[Any], list[AggregatedItem]]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: _RecordingQueue = _RecordingQueue()

        logger = logging.getLogger("test.enrichment.sample")
        logger.setLevel(logging.INFO)
        logger.handlers = [handler]
        logger.propagate = False

        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=IMDbRetryQueue(),
            movie_batch_size=2,
            episode_batch_size=2,
            logger=logger,
        )

        items = [
            AggregatedItem(
                plex=PlexItem(
                    rating_key="1",
                    guid="plex://1",
                    type="movie",
                    title="Sample 1",
                )
            ),
            AggregatedItem(
                plex=PlexItem(
                    rating_key="2",
                    guid="plex://2",
                    type="movie",
                    title="Sample 2",
                )
            ),
        ]

        await ingest_queue.put(SampleBatch(items=list(items)))
        await ingest_queue.put(INGEST_DONE)

        await stage.run()

        payloads: list[list[AggregatedItem] | None] = []
        while True:
            payload = await persistence_queue.get()
            payloads.append(payload)
            if payload in (None, PERSIST_DONE):
                break
        return payloads, persistence_queue.put_payloads, items

    payloads, put_payloads, items = asyncio.run(scenario())

    assert any("Processed sample batch" in message for message in handler.messages)
    assert len(payloads) == 2
    batch, sentinel = payloads
    assert sentinel is PERSIST_DONE
    assert isinstance(batch, list)
    assert batch == items
    assert put_payloads[0] == batch
    assert put_payloads[-1] is PERSIST_DONE


def test_enrichment_stage_retries_imdb_queue_when_idle(monkeypatch):
    calls: list[list[str]] = []

    async def fake_fetch_imdb_batch(client, imdb_ids, **kwargs):
        calls.append(list(imdb_ids))
        return {
            imdb_id: IMDbTitle(id=imdb_id, type="movie", primaryTitle=imdb_id)
            for imdb_id in imdb_ids
        }

    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_imdb_batch",
        fake_fetch_imdb_batch,
    )

    async def scenario() -> tuple[list[list[str]], int]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        retry_queue = IMDbRetryQueue(["tt1", "tt2"])

        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=retry_queue,
            movie_batch_size=2,
            episode_batch_size=2,
        )

        run_task = asyncio.create_task(stage.run())
        await asyncio.sleep(0)
        await ingest_queue.put(INGEST_DONE)
        await run_task
        return calls, retry_queue.qsize()

    processed, remaining = asyncio.run(scenario())
    assert processed == [["tt1", "tt2"]]
    assert remaining == 0


def test_enrichment_stage_idle_retry_emits_updated_items(monkeypatch):
    calls: list[list[str]] = []
    first_call = True

    async def fake_fetch_imdb_batch(client, imdb_ids, **kwargs):
        nonlocal first_call
        calls.append(list(imdb_ids))
        retry_queue: IMDbRetryQueue = kwargs["retry_queue"]
        if first_call:
            first_call = False
            for imdb_id in imdb_ids:
                retry_queue.put_nowait(imdb_id)
            return {imdb_id: None for imdb_id in imdb_ids}
        return {
            imdb_id: IMDbTitle(
                id=imdb_id,
                type="movie",
                primaryTitle=f"IMDb {imdb_id}",
            )
            for imdb_id in imdb_ids
        }

    monkeypatch.setattr(
        "mcp_plex.loader.pipeline.enrichment._fetch_imdb_batch",
        fake_fetch_imdb_batch,
    )

    async def scenario() -> tuple[list[list[AggregatedItem] | None], int, list[list[str]]]:
        ingest_queue: asyncio.Queue = asyncio.Queue()
        persistence_queue: asyncio.Queue = asyncio.Queue()
        retry_queue = IMDbRetryQueue()

        stage = EnrichmentStage(
            http_client_factory=lambda: object(),
            tmdb_api_key="",
            ingest_queue=ingest_queue,
            persistence_queue=persistence_queue,
            imdb_retry_queue=retry_queue,
            movie_batch_size=2,
            episode_batch_size=2,
        )

        run_task = asyncio.create_task(stage.run())
        await ingest_queue.put(MovieBatch(movies=[_FakeMovie("1", imdb_id="tt1")]))
        for _ in range(5):
            await asyncio.sleep(0)
            if persistence_queue.qsize() >= 2:
                break
        await ingest_queue.put(INGEST_DONE)
        await run_task

        payloads: list[list[AggregatedItem] | None] = []
        while True:
            payload = await persistence_queue.get()
            payloads.append(payload)
            if payload in (None, PERSIST_DONE):
                break
        return payloads, retry_queue.qsize(), calls

    payloads, remaining, captured_calls = asyncio.run(scenario())

    assert captured_calls == [["tt1"], ["tt1"]]
    assert remaining == 0
    assert len(payloads) == 3
    first_batch, second_batch, sentinel = payloads
    assert sentinel is PERSIST_DONE
    assert isinstance(first_batch, list)
    assert isinstance(second_batch, list)
    assert first_batch[0].imdb is None
    assert second_batch[0].imdb is not None
    assert second_batch[0].imdb.primaryTitle == "IMDb tt1"


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]):
        self.status_code = status_code
        self._payload = payload

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict[str, Any]:  # pragma: no cover - tiny helper
        return self._payload


def test_fetch_imdb_batch_success(monkeypatch):
    async def scenario() -> tuple[dict[str, IMDbTitle | None], int]:
        class _Client:
            def __init__(self) -> None:
                self.calls = 0

            async def get(self, url: str, params=None):
                self.calls += 1
                ids = [value for _, value in params]
                return _FakeResponse(
                    200,
                    {
                        "titles": [
                            {
                                "id": imdb_id,
                                "type": "movie",
                                "primaryTitle": imdb_id.upper(),
                            }
                            for imdb_id in ids
                        ]
                    },
                )

        client = _Client()
        throttle = _RequestThrottler(limit=10, interval=1.0)
        results = await _fetch_imdb_batch(
            client,
            ["tt1", "tt2"],
            cache=None,
            throttle=throttle,
            max_retries=2,
            backoff=0.01,
            retry_queue=None,
            batch_limit=5,
        )
        return results, client.calls

    results, calls = asyncio.run(scenario())
    assert calls == 1
    assert results["tt1"].primaryTitle == "TT1"
    assert results["tt2"].primaryTitle == "TT2"


def test_fetch_imdb_batch_rate_limit_retries(monkeypatch):
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def scenario() -> tuple[dict[str, IMDbTitle | None], int]:
        attempts = 0

        class _Client:
            async def get(self, url: str, params=None):
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    return _FakeResponse(429, {})
                return _FakeResponse(
                    200,
                    {
                        "titles": [
                            {
                                "id": params[0][1],
                                "type": "movie",
                                "primaryTitle": params[0][1],
                            }
                        ]
                    },
                )

        client = _Client()
        results = await _fetch_imdb_batch(
            client,
            ["tt3"],
            cache=None,
            throttle=None,
            max_retries=2,
            backoff=0.05,
            retry_queue=None,
            batch_limit=5,
        )
        return results, attempts

    results, attempts = asyncio.run(scenario())
    assert attempts == 2
    assert sleep_calls == [0.05]
    assert results["tt3"] is not None


def test_fetch_imdb_batch_rate_limit_exhaustion(monkeypatch):
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def scenario() -> tuple[dict[str, IMDbTitle | None], list[str]]:
        class _Client:
            async def get(self, url: str, params=None):
                return _FakeResponse(429, {})

        retry_queue = IMDbRetryQueue()
        results = await _fetch_imdb_batch(
            _Client(),
            ["tt4", "tt5"],
            cache=None,
            throttle=None,
            max_retries=1,
            backoff=0.01,
            retry_queue=retry_queue,
            batch_limit=5,
        )
        return results, retry_queue.snapshot()

    results, queued = asyncio.run(scenario())
    assert queued == ["tt4", "tt5"]
    assert all(value is None for value in results.values())
    assert sleep_calls == [0.01]
