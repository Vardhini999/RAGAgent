import pytest
import numpy as np
from src.cache.semantic_cache import SemanticCache


class FakeEmbedder:
    """Returns a fixed vector so we can control similarity."""
    def __init__(self, vector):
        self.vector = np.array(vector, dtype=float)

    def embed_one(self, text: str):
        return self.vector / np.linalg.norm(self.vector)


def make_cache(monkeypatch, vector):
    embedder = FakeEmbedder(vector)
    monkeypatch.setattr("src.cache.semantic_cache.embed_one", embedder.embed_one)
    return SemanticCache(threshold=0.9, max_size=10)


def test_cache_miss_on_empty(monkeypatch):
    cache = make_cache(monkeypatch, [1, 0, 0])
    assert cache.get("hello") is None
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == 0.0


def test_cache_hit_on_identical(monkeypatch):
    cache = make_cache(monkeypatch, [1, 0, 0])
    cache.set("hello", "world", "single")
    entry = cache.get("hello")
    assert entry is not None
    assert entry.response == "world"
    assert cache.stats.hits == 1


def test_cache_evicts_on_overflow(monkeypatch):
    cache = make_cache(monkeypatch, [1, 0, 0])
    cache.max_size = 2
    cache.set("a", "resp_a", "single")
    cache.set("b", "resp_b", "single")
    cache.set("c", "resp_c", "single")
    assert len(cache._store) == 2


def test_hit_rate_calculation(monkeypatch):
    cache = make_cache(monkeypatch, [1, 0, 0])
    cache.set("q", "r", "single")
    cache.get("q")   # hit
    cache.get("x")   # miss (same vector but store has "q", sim=1.0 ≥ 0.9 → hit actually)
    # Both hit because same embedding vector
    assert cache.stats.hits == 2
    assert cache.stats.hit_rate == 1.0
