"""Text embeddings and semantic search (cosine similarity)."""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder(name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed(model, texts: list[str]) -> np.ndarray:
    return np.asarray(model.encode(texts, normalize_embeddings=True, show_progress_bar=False))


class VectorIndex:
    """Tiny in-memory vector index. Vectors are normalised, so dot product = cosine similarity."""

    def __init__(self, model, chunks):
        self.model = model
        self.chunks = chunks
        self.vectors = embed(model, [c.text for c in chunks])

    def search(self, query: str, top_k: int = 3):
        q = embed(self.model, [query])[0]
        scores = self.vectors @ q
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]
