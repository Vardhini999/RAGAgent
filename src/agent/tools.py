"""Shared tools available to both single-agent and multi-agent modes."""
from langchain_core.tools import tool

from src.config import settings
from src.retrieval.hybrid import retrieve


@tool
def retrieve_context(query: str) -> str:
    """Retrieve relevant document chunks for a given query using hybrid search."""
    chunks = retrieve(query, k=settings.top_k)
    if not chunks:
        return "No relevant context found."
    parts = []
    for i, c in enumerate(chunks, 1):
        source = c["metadata"].get("source", "unknown")
        page = c["metadata"].get("page", "")
        loc = f"{source}" + (f" p.{page}" if page != "" else "")
        parts.append(f"[{i}] ({loc}, score={c['fused_score']})\n{c['text']}")
    return "\n\n".join(parts)


@tool
def assess_relevance(chunks_text: str, query: str) -> float:
    """Return a 0-1 relevance score for retrieved chunks relative to the query.
    Uses heuristic keyword overlap as a lightweight signal."""
    if not chunks_text or chunks_text == "No relevant context found.":
        return 0.0
    query_words = set(query.lower().split())
    chunk_words = set(chunks_text.lower().split())
    overlap = len(query_words & chunk_words)
    return min(1.0, overlap / max(len(query_words), 1))
