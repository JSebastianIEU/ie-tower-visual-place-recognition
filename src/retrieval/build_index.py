import faiss
import numpy as np

from src.retrieval.faiss_utils import create_index, ensure_float32, l2_normalize


def build_faiss_index(embeddings: np.ndarray, metric: str = "cosine") -> faiss.Index:
    embeddings = ensure_float32(embeddings)
    index = create_index(embedding_dim=embeddings.shape[1], metric=metric)

    if metric.lower() == "cosine":
        embeddings = l2_normalize(embeddings)

    index.add(embeddings)
    return index
