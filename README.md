# Agentic RAG — Document Intelligence Pipeline

> Domain-agnostic document Q&A with adaptive single/multi-agent routing, hybrid retrieval, semantic caching, LLM-as-judge evaluation, and full observability.

**Live demo:** https://vardhini999-raga.hf.space  
**Stack:** LangGraph · Groq (Llama 3.3 70B) · ChromaDB · BM25 · sentence-transformers · FastAPI · Gradio · Langfuse

---

## Architecture

```
User Query
    │
    ▼
Semantic Cache ──(hit)──────────────────────────────► Response
    │ (miss)
    ▼
Complexity Router  (Groq llama-3.1-8b, fast)
    │
    ├── score < 0.6 ──► Single Agent (ReAct + tools)
    │                        │
    └── score ≥ 0.6 ──► Multi-Agent Pipeline
                             ├── Supervisor  → decomposes query into sub-questions
                             ├── Retrieval Agent → hybrid search per sub-question
                             ├── Synthesis Agent → grounded answer with citations
                             └── Critique Agent  → removes unsupported claims

Both paths share:
  Hybrid Retrieval (BM25 sparse + vector dense → RRF fusion)
  Structured Pydantic outputs
  HITL interrupt on confidence < 0.4
  Langfuse trace (routing decision, tool calls, grader scores)
  LLM-as-judge grader (faithfulness + relevance) — async, non-blocking
```

---

## Baseline Comparisons

| Dimension | Baseline | This System |
|---|---|---|
| Retrieval | Vector-only | Hybrid BM25 + vector (RRF) |
| Agent | Single agent | Adaptive: single or multi-agent by complexity |
| Output format | Raw string | Pydantic-typed structured response |
| Low-confidence handling | None | HITL interrupt + human escalation flag |
| Observability | None | Langfuse: every step traced + scored |
| Repeated queries | Full LLM call | Semantic cache (cosine similarity threshold) |

---

## Key Metrics (run `python -m src.evaluation.offline_eval`)

Metrics populate after running the offline eval harness on your ingested documents.
Tracked per run: faithfulness, relevance, composite score, HITL escalation rate,
multi-agent routing rate, cache hit rate, p50/p95 latency.

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/Vardhini999/RAGAgent
cd agentic-rag
uv sync --extra dev

# 2. Set environment variables
cp .env.example .env
# Add your GROQ_API_KEY (free at console.groq.com)

# 3. Start the API
uv run uvicorn src.api.app:app --reload

# 4. Start the UI (separate terminal)
uv run python ui/app.py

# 5. Run offline eval
uv run python -m src.evaluation.offline_eval
```

---

## Project Structure

```
agentic-rag/
├── src/
│   ├── agent/
│   │   ├── graph.py          # LangGraph state machine
│   │   ├── router.py         # Complexity-based dispatcher
│   │   ├── single_agent.py   # ReAct agent with tools
│   │   ├── tools.py          # Shared retrieve + assess tools
│   │   ├── state.py          # Typed AgentState (Pydantic)
│   │   └── multi_agent/
│   │       ├── supervisor.py       # Query decomposition
│   │       ├── retrieval_agent.py  # Per-sub-question hybrid search
│   │       ├── synthesis_agent.py  # Grounded answer generation
│   │       └── critique_agent.py   # Hallucination filtering
│   ├── retrieval/
│   │   ├── chunker.py        # PDF/TXT → chunks
│   │   ├── embedder.py       # sentence-transformers (local)
│   │   ├── vectorstore.py    # ChromaDB (swappable interface)
│   │   ├── bm25.py           # BM25 sparse retrieval
│   │   └── hybrid.py         # RRF fusion
│   ├── cache/
│   │   └── semantic_cache.py # Cosine-similarity LRU cache
│   ├── evaluation/
│   │   ├── grader.py         # LLM-as-judge (async)
│   │   └── offline_eval.py   # Ground-truth eval harness
│   ├── guardrails/
│   │   └── guardrails.py     # Input/output safety checks
│   ├── api/
│   │   ├── app.py            # FastAPI app
│   │   ├── schemas.py        # Pydantic request/response models
│   │   └── routes/
│   │       ├── ingest.py     # POST /ingest
│   │       └── query.py      # POST /query, GET /cache/stats
│   ├── observability.py      # Langfuse wrapper (no-op if key absent)
│   └── config.py             # Pydantic settings (env-based)
├── ui/app.py                 # Gradio UI
├── eval_data/test_set.json   # Ground-truth eval examples
├── tests/                    # pytest suite
└── .github/workflows/ci.yml  # GitHub Actions: lint + test
```

---

## Resume Bullets

- Designed and deployed an **adaptive agentic RAG system** using LangGraph, routing queries to single-agent (ReAct) or multi-agent (Supervisor → Retrieval → Synthesis → Critique) mode based on a complexity score — demonstrating intentional architectural tradeoff between speed and reasoning depth
- Built **hybrid retrieval** (BM25 sparse + vector dense with RRF fusion) over ChromaDB, improving recall on keyword-heavy queries vs. vector-only baseline; retrieval layer abstracted for drop-in swap to Qdrant/Pinecone at scale
- Implemented **semantic caching** with cosine-similarity threshold, reducing redundant LLM calls and improving p50 latency on repeated/similar queries
- Wired **end-to-end observability** with Langfuse — every routing decision, tool call, retrieval step, and grader score traced in production; **LLM-as-judge grader** (faithfulness + relevance) runs async, non-blocking
- Applied **human-in-the-loop** interrupts for low-confidence responses and **input/output guardrails** for safe production deployment
- Shipped **offline eval harness** with ground-truth test set measuring faithfulness, relevance, HITL escalation rate, and agent mode distribution across query types
- Hosted on **Hugging Face Spaces** with FastAPI backend; **GitHub Actions** CI runs ruff lint + pytest on every push
