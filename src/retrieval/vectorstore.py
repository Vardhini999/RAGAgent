from __future__ import annotations

import chromadb
from langchain_core.documents import Document
from src.config import settings
from src.retrieval.embedder import embed
import uuid

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_path))
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(docs: list[Document]) -> int:
    col = _get_collection()
    texts = [d.page_content for d in docs]
    embeddings = embed(texts).tolist()
    ids = [str(uuid.uuid4()) for _ in docs]
    metadatas = [d.metadata for d in docs]
    col.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(docs)


def similarity_search(query: str, k: int | None = None) -> list[dict]:
    col = _get_collection()
    k = k or settings.top_k
    query_embedding = embed([query])[0].tolist()
    results = col.query(query_embeddings=[query_embedding], n_results=min(k, col.count() or 1))
    chunks = []
    for i, text in enumerate(results["documents"][0]):
        chunks.append({
            "text": text,
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],  # cosine similarity
        })
    return chunks


def collection_count() -> int:
    return _get_collection().count()


def reset_collection() -> None:
    global _collection
    if _client is not None:
        _client.delete_collection(settings.chroma_collection)
    _collection = None
    _get_collection()
