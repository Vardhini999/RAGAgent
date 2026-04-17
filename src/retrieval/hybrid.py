"""
Hybrid retrieval using Reciprocal Rank Fusion (RRF) over vector + BM25 results.
alpha controls vector weight: 1.0 = pure vector, 0.0 = pure BM25.
"""
from src.config import settings
from src.retrieval import bm25, vectorstore

_RRF_K = 60  # standard RRF constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (_RRF_K + rank + 1)


def retrieve(query: str, k: int | None = None) -> list[dict]:
    k = k or settings.top_k
    fetch_k = k * 2  # over-fetch before fusion

    vector_results = vectorstore.similarity_search(query, k=fetch_k)
    bm25_results = bm25.search(query, k=fetch_k)

    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, item in enumerate(vector_results):
        key = item["text"][:120]
        scores[key] = scores.get(key, 0) + settings.hybrid_alpha * _rrf_score(rank)
        payloads[key] = item

    for rank, item in enumerate(bm25_results):
        key = item["text"][:120]
        scores[key] = scores.get(key, 0) + (1 - settings.hybrid_alpha) * _rrf_score(rank)
        if key not in payloads:
            payloads[key] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    results = []
    for key, fused_score in ranked:
        item = dict(payloads[key])
        item["fused_score"] = round(fused_score, 4)
        results.append(item)
    return results
