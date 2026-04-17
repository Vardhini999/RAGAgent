"""
Single-agent mode: LangGraph prebuilt ReAct agent with retrieve + assess tools.
Used for focused, low-complexity queries.
"""
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from src.agent.tools import retrieve_context
from src.agent.state import AgentState
from src.config import settings
from src.retrieval.hybrid import retrieve


def run(state: AgentState) -> AgentState:
    llm = ChatGroq(api_key=settings.groq_api_key, model=settings.groq_model, temperature=0.1)
    tools = [retrieve_context]

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=(
            "You are a precise question-answering agent with access to a document knowledge base.\n"
            "Always retrieve context before answering document-based questions.\n"
            "If retrieved context is not relevant, try once more with a reformulated query.\n"
            "Cite your sources inline. If you cannot find the answer, say so clearly."
        ),
    )

    result = agent.invoke({"messages": [HumanMessage(content=state.query)]})
    last_msg = result["messages"][-1]
    answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    chunks = retrieve(state.query, k=settings.top_k)
    citations = list({c["metadata"].get("source", "unknown") for c in chunks})
    confidence = 0.85 if chunks else 0.3

    state.answer = answer
    state.citations = citations
    state.confidence = confidence
    state.mode = "single"
    state.retrieved_chunks = chunks
    return state
