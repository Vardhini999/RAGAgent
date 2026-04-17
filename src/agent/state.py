from typing import Annotated, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class AgentState(BaseModel):
    """Shared state flowing through the LangGraph graph."""
    messages: Annotated[list, add_messages] = Field(default_factory=list)
    query: str = ""
    mode: Literal["single", "multi", "cache"] = "single"
    retrieved_chunks: list[dict] = Field(default_factory=list)
    retrieval_attempts: int = 0
    answer: str = ""
    confidence: float = 1.0
    citations: list[str] = Field(default_factory=list)
    hitl_required: bool = False
    grader_score: float | None = None
    trace_id: str = ""
    cache_hit: bool = False
