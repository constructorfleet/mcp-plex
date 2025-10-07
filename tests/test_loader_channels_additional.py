from __future__ import annotations

import asyncio

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
