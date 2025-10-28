from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client import models
import pytest

from mcp_plex import loader


class CaptureClient(AsyncQdrantClient):
    instance: "CaptureClient" | None = None
    captured_points: list[models.PointStruct] = []
    upsert_calls: int = 0
    created_indexes: list[tuple[str, Any]] = []
    close_calls: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CaptureClient.instance = self

    async def upsert(self, collection_name: str, points, **kwargs):
        CaptureClient.upsert_calls += 1
        CaptureClient.captured_points.extend(points)
        return await super().upsert(
            collection_name=collection_name, points=points, **kwargs
        )

    async def create_payload_index(
        self,
        collection_name: str,
        field_name: str,
        field_schema: models.PayloadSchemaType | models.TextIndexParams,
        wait: bool | None = None,
    ) -> models.UpdateResult:
        CaptureClient.created_indexes.append((field_name, field_schema))
        return await super().create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=field_schema,
            wait=wait,
        )

    async def close(self) -> None:
        CaptureClient.close_calls += 1
        await super().close()


async def _run_loader(sample_dir: Path, **kwargs) -> None:
    await loader.run(
        None,
        None,
        None,
        sample_dir,
        None,
        None,
        **kwargs,
    )


def test_run_writes_points(monkeypatch):
    monkeypatch.setattr(loader, "AsyncQdrantClient", CaptureClient)
    CaptureClient.captured_points = []
    CaptureClient.upsert_calls = 0
    CaptureClient.created_indexes = []
    CaptureClient.close_calls = 0
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    asyncio.run(_run_loader(sample_dir))
    client = CaptureClient.instance
    assert client is not None
    index_map = {name: schema for name, schema in CaptureClient.created_indexes}
    assert index_map.get("show_title") == models.PayloadSchemaType.KEYWORD
    assert index_map.get("season_number") == models.PayloadSchemaType.INTEGER
    assert index_map.get("episode_number") == models.PayloadSchemaType.INTEGER
    captured = CaptureClient.captured_points
    assert len(captured) == 2
    assert all(isinstance(p.vector["dense"], models.Document) for p in captured)
    assert all(p.vector["dense"].model == "BAAI/bge-small-en-v1.5" for p in captured)
    assert all(isinstance(p.vector["sparse"], models.Document) for p in captured)
    assert all(
        p.vector["sparse"].model == "Qdrant/bm42-all-minilm-l6-v2-attentions"
        for p in captured
    )
    texts = [p.vector["dense"].text for p in captured]
    assert any("Directed by" in t for t in texts)
    assert any("Starring" in t for t in texts)
    movie_point = next(p for p in captured if p.payload["type"] == "movie")
    assert (
        "directors" in movie_point.payload
        and "Guy Ritchie" in movie_point.payload["directors"]
    )
    assert "writers" in movie_point.payload and movie_point.payload["writers"]
    assert "genres" in movie_point.payload and movie_point.payload["genres"]
    assert movie_point.payload.get("summary")
    assert movie_point.payload.get("overview")
    assert movie_point.payload.get("plot")
    assert movie_point.payload.get("tagline")
    assert movie_point.payload.get("reviews")
    episode_point = next(p for p in captured if p.payload["type"] == "episode")
    assert episode_point.payload.get("show_title") == "Alien: Earth"
    assert episode_point.payload.get("season_title") == "Season 1"
    assert episode_point.payload.get("season_number") == 1
    assert episode_point.payload.get("episode_number") == 4
    episode_vector = (
        next(p for p in captured if p.payload.get("type") == "episode")
        .vector["dense"]
        .text
    )
    assert "Alien: Earth" in episode_vector
    assert "S01E04" in episode_vector


def test_run_processes_imdb_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(loader, "AsyncQdrantClient", CaptureClient)
    CaptureClient.captured_points = []
    CaptureClient.upsert_calls = 0
    CaptureClient.close_calls = 0
    queue_file = tmp_path / "queue.json"
    queue_file.write_text(json.dumps(["tt0111161"]))
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"

    async def fake_fetch(client, imdb_id, config):
        return None

    monkeypatch.setattr(loader, "_fetch_imdb", fake_fetch)

    asyncio.run(
        loader.run(
            None,
            None,
            None,
            sample_dir,
            None,
            None,
            imdb_queue_path=queue_file,
        )
    )

    assert json.loads(queue_file.read_text()) == ["tt0111161"]


def test_run_upserts_in_batches(monkeypatch):
    monkeypatch.setattr(loader, "AsyncQdrantClient", CaptureClient)
    CaptureClient.captured_points = []
    CaptureClient.upsert_calls = 0
    CaptureClient.close_calls = 0
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"
    asyncio.run(_run_loader(sample_dir, qdrant_batch_size=1))
    assert CaptureClient.upsert_calls == 2
    assert len(CaptureClient.captured_points) == 2


def test_run_closes_client_once(monkeypatch):
    monkeypatch.setattr(loader, "AsyncQdrantClient", CaptureClient)
    CaptureClient.close_calls = 0
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"

    asyncio.run(_run_loader(sample_dir))
    assert CaptureClient.close_calls == 1

    asyncio.run(_run_loader(sample_dir))
    assert CaptureClient.close_calls == 2


def test_run_skips_existing_media(monkeypatch):
    class ExistingClient(CaptureClient):
        retrieved_ids: list[tuple[int | str, ...]] = []

        async def retrieve(self, collection_name: str, ids, **kwargs):
            ExistingClient.retrieved_ids.append(tuple(ids))

            class _Record:
                def __init__(self, point_id):
                    self.id = point_id

            return [_Record(point_id=id_value) for id_value in ids]

    monkeypatch.setattr(loader, "AsyncQdrantClient", ExistingClient)
    CaptureClient.captured_points = []
    CaptureClient.upsert_calls = 0
    ExistingClient.retrieved_ids = []

    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"

    asyncio.run(_run_loader(sample_dir))

    assert CaptureClient.captured_points == []
    assert CaptureClient.upsert_calls == 0
    assert ExistingClient.retrieved_ids


def test_run_raises_for_unknown_dense_model():
    sample_dir = Path(__file__).resolve().parents[1] / "sample-data"

    with pytest.raises(ValueError, match="Unknown dense embedding model"):
        asyncio.run(
            loader.run(
                None,
                None,
                None,
                sample_dir,
                None,
                None,
                dense_model_name="not-a-real/model",
            )
        )
