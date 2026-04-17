from langchain_core.documents import Document

from src.retrieval import bm25, hybrid
from src.retrieval.chunker import load_and_chunk_bytes


def make_doc(text: str, source: str = "test.txt") -> Document:
    return Document(page_content=text, metadata={"source": source, "chunk_id": 0})


def test_bm25_search_returns_results():
    docs = [
        make_doc("The capital of France is Paris."),
        make_doc("Python is a programming language."),
    ]
    bm25.build_index(docs)
    results = bm25.search("What is the capital of France?", k=2)
    assert len(results) >= 1
    assert "Paris" in results[0]["text"]


def test_bm25_empty_index_returns_empty():
    bm25.build_index([])
    results = bm25.search("anything", k=3)
    assert results == []


def test_chunker_bytes():
    content = b"This is a test document.\n\nIt has multiple paragraphs.\n\nEach paragraph is a chunk."  # noqa: E501
    chunks = load_and_chunk_bytes(content, "test.txt")
    assert len(chunks) >= 1
    assert all(c.metadata["source"] == "test.txt" for c in chunks)


def test_hybrid_rrf_deduplicates(monkeypatch):
    chunk = {"text": "Paris is the capital of France.", "metadata": {"source": "test.txt"}, "fused_score": 0.5}  # noqa: E501

    monkeypatch.setattr("src.retrieval.hybrid.vectorstore.similarity_search", lambda q, k: [chunk])
    monkeypatch.setattr("src.retrieval.hybrid.bm25.search", lambda q, k: [chunk])

    results = hybrid.retrieve("capital of France", k=3)
    texts = [r["text"] for r in results]
    assert texts.count(chunk["text"]) == 1
