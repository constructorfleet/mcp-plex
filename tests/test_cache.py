from mcp_plex.cache import MediaCache


def test_media_cache_eviction_and_clear():
    cache = MediaCache(size=2)
    cache.set_payload("a", {"id": 1})
    cache.set_payload("b", {"id": 2})
    cache.get_payload("a")
    cache.set_payload("c", {"id": 3})
    assert cache.get_payload("a") == {"id": 1}
    assert cache.get_payload("b") is None
    assert cache.get_payload("c") == {"id": 3}

    assert cache.get_poster("missing") is None
    cache.set_poster("p1", "poster")
    cache.set_background("bg1", "background")
    cache.clear()
    assert cache.get_payload("a") is None
    assert cache.get_poster("p1") is None
    assert cache.get_background("bg1") is None
