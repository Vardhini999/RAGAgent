"""
Critique agent: reviews the synthesized answer for factual grounding,
completeness, and hallucination risk. Returns a revised answer + final confidence.
"""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from src.config import settings

_SYSTEM = """You are a critique agent reviewing an AI-generated answer.

Given the original query, the retrieved context, and the draft answer:
1. Check every claim in the answer is supported by the context
2. Flag unsupported claims and remove or qualify them
3. Check the answer fully addresses the query
4. Output the revised answer followed by a confidence score on a new line as:
   CONFIDENCE: 0.XX

Be concise. If the draft answer is already correct, output it unchanged with a confidence score."""


def run(query: str, draft_answer: str, chunks: list[dict]) -> tuple[str, float]:
    context = "\n\n".join(c["text"] for c in chunks[:6])
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model_fast, temperature=0)
    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=(
            f"Query: {query}\n\n"
            f"Context:\n{context}\n\n"
            f"Draft Answer:\n{draft_answer}"
        )),
    ])
    text = response.content
    confidence = 0.75  # default
    match = re.search(r"CONFIDENCE:\s*([0-9.]+)", text)
    if match:
        confidence = float(match.group(1))
        text = text[: match.start()].strip()
    return text, round(confidence, 2)
