"""
Complexity router: decides single-agent vs multi-agent mode.
Uses a fast LLM call (groq_model_fast) to score query complexity 0-1.
Above complexity_threshold → multi-agent.
Also checks semantic cache before routing.
"""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.cache.semantic_cache import get_cache
from src.config import settings

_SYSTEM = """You are a query complexity classifier. Given a user query, output a single float
between 0.0 and 1.0 representing its complexity for a retrieval-augmented system.

0.0 = trivial factual lookup, single document chunk sufficient
0.5 = moderate, needs synthesis across multiple chunks
1.0 = complex multi-step reasoning, comparison, or multi-document analysis

Respond with ONLY the float. No explanation."""


def _parse_score(text: str) -> float:
    match = re.search(r"\d+\.?\d*", text)
    return float(match.group()) if match else 0.5


def route(query: str) -> dict:
    """Returns {"mode": "cache"|"single"|"multi", "cache_entry": ..., "complexity": float}"""
    cache = get_cache()
    cached = cache.get(query)
    if cached:
        return {"mode": "cache", "cache_entry": cached, "complexity": 0.0}

    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model_fast, temperature=0)
    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=query),
    ])
    complexity = min(1.0, max(0.0, _parse_score(response.content)))
    mode = "multi" if complexity >= settings.complexity_threshold else "single"
    return {"mode": mode, "cache_entry": None, "complexity": complexity}
