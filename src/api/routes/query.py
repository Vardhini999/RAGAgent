import asyncio
import uuid

from fastapi import APIRouter, HTTPException

from src.agent.graph import run_query
from src.api.schemas import QueryRequest, QueryResponse
from src.cache.semantic_cache import get_cache
from src.evaluation.grader import grade_async
from src.guardrails.guardrails import GuardrailViolation, check_input, check_output
from src.observability import Trace

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        safe_query = check_input(request.query)
    except GuardrailViolation as e:
        raise HTTPException(status_code=400, detail=str(e))

    trace_id = str(uuid.uuid4())[:8]

    with Trace("query", metadata={"query": safe_query, "trace_id": trace_id}) as trace:
        try:
            state = await asyncio.get_event_loop().run_in_executor(
                None, run_query, safe_query, trace_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Agent error: {e}")

        answer = check_output(state.answer)

        trace.span("routing", input=safe_query, output=state.mode)
        trace.span("retrieval", input=safe_query, output=f"{len(state.retrieved_chunks)} chunks")
        trace.span("answer", input=safe_query, output=answer[:200])

        # Fire grader async — doesn't block response
        grader_score = None
        if not state.cache_hit and state.retrieved_chunks:
            context = "\n".join(c["text"] for c in state.retrieved_chunks[:3])
            try:
                result = await grade_async(safe_query, answer, context)
                grader_score = result.composite
                trace.score("composite", grader_score)
                trace.score("faithfulness", result.faithfulness)
                trace.score("relevance", result.relevance)
            except Exception:
                pass

        trace.span("cache", input="hit", output=str(state.cache_hit))

    return QueryResponse(
        answer=answer,
        mode=state.mode,
        citations=state.citations,
        confidence=state.confidence,
        cache_hit=state.cache_hit,
        hitl_required=state.hitl_required,
        grader_score=grader_score,
        trace_id=trace_id,
    )


@router.get("/cache/stats")
async def cache_stats():
    stats = get_cache().stats
    return {"hits": stats.hits, "misses": stats.misses, "hit_rate": stats.hit_rate}
