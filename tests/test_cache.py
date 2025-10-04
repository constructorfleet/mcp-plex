from mcp_plex.common import MediaCache


def test_media_cache_eviction_and_clear():
    cache = MediaCache(size=2)
    cache.set_payload(
        "tt0111161", {"id": "tt0111161", "title": "The Shawshank Redemption"}
    )
    cache.set_payload("tt0068646", {"id": "tt0068646", "title": "The Godfather"})
    cache.get_payload("tt0111161")
    cache.set_payload("tt1375666", {"id": "tt1375666", "title": "Inception"})
    assert cache.get_payload("tt0111161") == {
        "id": "tt0111161",
        "title": "The Shawshank Redemption",
    }
    assert cache.get_payload("tt0068646") is None
    assert cache.get_payload("tt1375666") == {
        "id": "tt1375666",
        "title": "Inception",
    }

    assert cache.get_poster("missing") is None
    cache.set_poster("tt0111161", "https://example.com/shawshank.jpg")
    cache.set_background("tt0111161", "https://example.com/shawshank-bg.jpg")
    assert cache.get_poster("tt0111161") == "https://example.com/shawshank.jpg"
    assert cache.get_background("tt0111161") == "https://example.com/shawshank-bg.jpg"
    cache.clear()
    assert cache.get_payload("tt0111161") is None
    assert cache.get_poster("tt0111161") is None
    assert cache.get_background("tt0111161") is None
