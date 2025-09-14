import asyncio
import logging

from mcp_plex import loader


async def _echo(value: int) -> int:
    await asyncio.sleep(0)
    return value


def test_gather_in_batches(monkeypatch, caplog):
    calls: list[int] = []
    orig_gather = asyncio.gather

    async def fake_gather(*coros):
        calls.append(len(coros))
        return await orig_gather(*coros)

    monkeypatch.setattr(asyncio, "gather", fake_gather)

    tasks = [_echo(i) for i in range(5)]
    with caplog.at_level(logging.INFO, logger="mcp_plex.loader"):
        results = asyncio.run(loader._gather_in_batches(tasks, 2))

    assert results == list(range(5))
    assert calls == [2, 2, 1]
    assert "Processed 2/5 items" in caplog.text
    assert "Processed 4/5 items" in caplog.text
    assert "Processed 5/5 items" in caplog.text

