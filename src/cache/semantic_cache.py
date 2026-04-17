"""
Semantic cache: stores (query_embedding, response) pairs.
Cache hit when cosine similarity between new query and a cached query exceeds threshold.
Tracks hit/miss counts for metrics reporting.
"""
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from src.config import settings
from src.retrieval.embedder import embed_one


@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    response: str
    mode_used: str


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0


class SemanticCache:
    def __init__(
        self,
        threshold: float = settings.cache_similarity_threshold,
        max_size: int = settings.cache_max_size,
    ):
        self.threshold = threshold
        self.max_size = max_size
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # embeddings are already L2-normalized

    def get(self, query: str) -> CacheEntry | None:
        if not self._store:
            self.stats.misses += 1
            return None
        query_emb = embed_one(query)
        best_sim, best_entry = 0.0, None
        for entry in self._store.values():
            sim = self._cosine_sim(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim, best_entry = sim, entry
        if best_sim >= self.threshold:
            self.stats.hits += 1
            return best_entry
        self.stats.misses += 1
        return None

    def set(self, query: str, response: str, mode_used: str) -> None:
        if len(self._store) >= self.max_size:
            self._store.popitem(last=False)  # evict oldest (LRU)
        embedding = embed_one(query)
        self._store[query] = CacheEntry(
            query=query, embedding=embedding, response=response, mode_used=mode_used
        )

    def clear(self) -> None:
        self._store.clear()
        self.stats = CacheStats()


# Module-level singleton
_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
