from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import re

_index: BM25Okapi | None = None
_corpus: list[str] = []
_docs: list[Document] = []


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_index(docs: list[Document]) -> None:
    global _index, _corpus, _docs
    _docs = docs
    _corpus = [d.page_content for d in docs]
    if not _corpus:
        _index = None
        return
    tokenized = [_tokenize(t) for t in _corpus]
    _index = BM25Okapi(tokenized)


def search(query: str, k: int) -> list[dict]:
    if _index is None or not _docs:
        return []
    scores = _index.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    results = []
    for idx in top_indices:
        results.append({
            "text": _corpus[idx],
            "metadata": _docs[idx].metadata,
            "score": float(scores[idx]),
        })
    return results


def get_all_docs() -> list[Document]:
    return _docs
