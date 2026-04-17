"""
LLM-as-judge grader: scores a response on faithfulness and answer relevance.
Runs async so it never blocks the response returned to the user.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from src.config import settings
import re
import asyncio

_SYSTEM = """You are an evaluation judge for a RAG system.
Score the response on two axes (0.0 - 1.0):

FAITHFULNESS: Is every claim in the response grounded in the provided context?
  1.0 = fully grounded, 0.0 = major hallucinations present

RELEVANCE: Does the response directly and completely answer the query?
  1.0 = fully answers query, 0.0 = does not address query

Output exactly:
FAITHFULNESS: 0.XX
RELEVANCE: 0.XX"""


class GraderResult(BaseModel):
    faithfulness: float
    relevance: float

    @property
    def composite(self) -> float:
        return round((self.faithfulness + self.relevance) / 2, 3)


def _parse(text: str) -> GraderResult:
    f_match = re.search(r"FAITHFULNESS:\s*([0-9.]+)", text)
    r_match = re.search(r"RELEVANCE:\s*([0-9.]+)", text)
    return GraderResult(
        faithfulness=float(f_match.group(1)) if f_match else 0.5,
        relevance=float(r_match.group(1)) if r_match else 0.5,
    )


def grade(query: str, response: str, context: str) -> GraderResult:
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model_fast, temperature=0)
    result = llm.invoke([
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=(
            f"Query: {query}\n\n"
            f"Context (retrieved):\n{context[:2000]}\n\n"
            f"Response:\n{response}"
        )),
    ])
    return _parse(result.content)


async def grade_async(query: str, response: str, context: str) -> GraderResult:
    return await asyncio.get_event_loop().run_in_executor(
        None, grade, query, response, context
    )
