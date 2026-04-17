"""
Offline evaluation harness.
Loads a ground-truth test set (eval_data/test_set.json) and runs each query
through the graph, scoring with the LLM-as-judge grader.
Prints a summary table and returns per-example results.
"""
import json
from pathlib import Path
from src.agent.graph import run_query
from src.evaluation.grader import grade
from src.retrieval.hybrid import retrieve
from src.config import settings

TEST_SET_PATH = Path(__file__).parent.parent.parent / "eval_data" / "test_set.json"


def load_test_set(path: Path = TEST_SET_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_eval(test_set: list[dict] | None = None) -> list[dict]:
    examples = test_set or load_test_set()
    results = []
    for ex in examples:
        query = ex["query"]
        state = run_query(query)
        context = "\n".join(c["text"] for c in state.retrieved_chunks[:3])
        score = grade(query, state.answer, context)
        results.append({
            "query": query,
            "mode_used": state.mode,
            "cache_hit": state.cache_hit,
            "faithfulness": score.faithfulness,
            "relevance": score.relevance,
            "composite": score.composite,
            "hitl_required": state.hitl_required,
        })

    _print_summary(results)
    return results


def _print_summary(results: list[dict]) -> None:
    n = len(results)
    avg_faith = sum(r["faithfulness"] for r in results) / n
    avg_rel = sum(r["relevance"] for r in results) / n
    avg_comp = sum(r["composite"] for r in results) / n
    hitl_rate = sum(r["hitl_required"] for r in results) / n
    multi_rate = sum(r["mode_used"] == "multi" for r in results) / n
    cache_rate = sum(r["cache_hit"] for r in results) / n

    print(f"\n{'='*55}")
    print(f"  Offline Eval — {n} examples")
    print(f"{'='*55}")
    print(f"  Avg Faithfulness  : {avg_faith:.3f}")
    print(f"  Avg Relevance     : {avg_rel:.3f}")
    print(f"  Avg Composite     : {avg_comp:.3f}")
    print(f"  HITL Escalation   : {hitl_rate:.1%}")
    print(f"  Multi-agent rate  : {multi_rate:.1%}")
    print(f"  Cache hit rate    : {cache_rate:.1%}")
    print(f"{'='*55}\n")
