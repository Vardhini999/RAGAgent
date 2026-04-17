"""
Main LangGraph graph.
Flow:
  START → route_node → [cache_node | single_node | multi_node] → hitl_check → END
"""
from langgraph.graph import END, StateGraph

from src.agent import single_agent
from src.agent.multi_agent import critique_agent, retrieval_agent, supervisor, synthesis_agent
from src.agent.router import route
from src.agent.state import AgentState
from src.cache.semantic_cache import get_cache
from src.config import settings

# ── Node functions ────────────────────────────────────────────────────────────

def route_node(state: AgentState) -> AgentState:
    result = route(state.query)
    state.mode = result["mode"]
    if result["mode"] == "cache":
        entry = result["cache_entry"]
        state.answer = entry.response
        state.cache_hit = True
        state.confidence = 1.0
    return state


def cache_node(state: AgentState) -> AgentState:
    return state  # answer already set by route_node


def single_node(state: AgentState) -> AgentState:
    return single_agent.run(state)


def multi_node(state: AgentState) -> AgentState:
    sub_questions = supervisor.decompose(state.query)
    chunks = retrieval_agent.run(sub_questions)
    draft, conf = synthesis_agent.run(state.query, chunks)
    answer, confidence = critique_agent.run(state.query, draft, chunks)
    citations = list({c["metadata"].get("source", "unknown") for c in chunks})
    state.answer = answer
    state.confidence = confidence
    state.citations = citations
    state.retrieved_chunks = chunks
    state.mode = "multi"
    return state


def hitl_check_node(state: AgentState) -> AgentState:
    if state.confidence < settings.hitl_confidence_threshold:
        state.hitl_required = True
        state.answer = (
            f"[Low confidence — human review requested]\n\n{state.answer}"
        )
    if not state.cache_hit:
        get_cache().set(state.query, state.answer, state.mode)
    return state


# ── Conditional edges ─────────────────────────────────────────────────────────

def dispatch(state: AgentState) -> str:
    return state.mode  # "cache" | "single" | "multi"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("route", route_node)
    builder.add_node("cache", cache_node)
    builder.add_node("single", single_node)
    builder.add_node("multi", multi_node)
    builder.add_node("hitl_check", hitl_check_node)

    builder.set_entry_point("route")
    builder.add_conditional_edges("route", dispatch, {
        "cache": "cache",
        "single": "single",
        "multi": "multi",
    })
    builder.add_edge("cache", "hitl_check")
    builder.add_edge("single", "hitl_check")
    builder.add_edge("multi", "hitl_check")
    builder.add_edge("hitl_check", END)

    return builder.compile()


graph = build_graph()


def run_query(query: str, trace_id: str = "") -> AgentState:
    initial = AgentState(query=query, trace_id=trace_id)
    result = graph.invoke(initial)
    return AgentState(**result)
