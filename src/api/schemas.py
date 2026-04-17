from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks_in_store: int


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str
    mode: Literal["single", "multi", "cache"]
    citations: list[str]
    confidence: float
    cache_hit: bool
    hitl_required: bool
    grader_score: float | None = None
    trace_id: str = ""
