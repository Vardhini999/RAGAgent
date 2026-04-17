"""
Supervisor agent: decomposes the query into sub-tasks and delegates to
retrieval_agent → synthesis_agent → critique_agent in sequence.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import settings

_SYSTEM = """You are a supervisor orchestrating a multi-agent RAG pipeline.
Given a complex user query, decompose it into 2-4 focused sub-questions
that together answer the full query. Return them as a numbered list only."""


def decompose(query: str) -> list[str]:
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model_fast, temperature=0)
    response = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Query: {query}"),
    ])
    lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
    sub_questions = []
    for line in lines:
        # strip leading number/punctuation
        cleaned = line.lstrip("0123456789.-) ").strip()
        if cleaned:
            sub_questions.append(cleaned)
    return sub_questions or [query]
