"""Synthesis agent: produces a structured, cited answer from retrieved chunks."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import settings

_SYSTEM = """You are a synthesis agent. You receive a user query and a set of retrieved
document chunks. Produce a clear, well-structured answer that:
1. Directly addresses the query
2. Draws on ALL relevant chunks
3. Cites sources inline as [source_name]
4. Flags uncertainty explicitly when evidence is weak

Return your answer as plain text."""


def run(query: str, chunks: list[dict]) -> tuple[str, float]:
    """Returns (answer, confidence_score)."""
    if not chunks:
        return "I could not find relevant information to answer this query.", 0.1

    context_parts = []
    for i, c in enumerate(chunks, 1):
        src = c["metadata"].get("source", "unknown")
        context_parts.append(f"[{i}] [{src}]\n{c['text']}")
    context = "\n\n".join(context_parts)

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, temperature=0.1)
    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Query: {query}\n\nContext:\n{context}"),
    ])
    answer = response.content

    # Heuristic confidence: proportion of chunks cited in answer
    cited = sum(
        1 for c in chunks
        if c["metadata"].get("source", "")[:15] in answer
    )
    confidence = round(0.5 + 0.5 * (cited / max(len(chunks), 1)), 2)
    return answer, confidence
