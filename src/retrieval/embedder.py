from sentence_transformers import SentenceTransformer
from src.config import settings
import numpy as np

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embed_model)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    return get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
