"""Retrieval agent: runs hybrid search for each sub-question, deduplicates chunks."""
from src.config import settings
from src.retrieval.hybrid import retrieve


def run(sub_questions: list[str], k: int | None = None) -> list[dict]:
    k = k or settings.top_k
    seen_texts: set[str] = set()
    all_chunks: list[dict] = []
    for question in sub_questions:
        for chunk in retrieve(question, k=k):
            key = chunk["text"][:80]
            if key not in seen_texts:
                seen_texts.add(key)
                chunk["sub_question"] = question
                all_chunks.append(chunk)
    return all_chunks
