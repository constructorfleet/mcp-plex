from __future__ import annotations

import asyncio
from typing import Iterable

from mcp_plex.loader.pipeline import channels


def test_enqueue_nowait_applies_backpressure():
    async def _run():
        queue: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
        await queue.put(1)

        enqueue_task = asyncio.create_task(channels.enqueue_nowait(queue, 2))
        await asyncio.sleep(0)
        assert not enqueue_task.done()

        await queue.get()
        await enqueue_task

        assert queue.qsize() == 1
        assert queue.get_nowait() == 2

    asyncio.run(_run())


def test_chunk_sequence_accepts_iterable_without_length():
    produced: list[int] = []

    async def _run() -> list[list[int]]:
        def _numbers() -> Iterable[int]:
            for value in range(5):
                produced.append(value)
                yield value

        chunks = [
            list(chunk) for chunk in channels.chunk_sequence(_numbers(), size=2)
        ]
        return chunks

    chunks = asyncio.run(_run())

    assert chunks == [[0, 1], [2, 3], [4]]
    assert produced == [0, 1, 2, 3, 4]
