from pathlib import Path

import faiss
import numpy as np


def ensure_float32(vectors: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    normalized = ensure_float32(vectors).copy()
    faiss.normalize_L2(normalized)
    return normalized


def create_index(embedding_dim: int, metric: str = "cosine") -> faiss.Index:
    metric = metric.lower()

    if metric == "cosine":
        return faiss.IndexFlatIP(embedding_dim)
    if metric == "l2":
        return faiss.IndexFlatL2(embedding_dim)

    raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'l2'.")


def save_index(index: faiss.Index, index_path: str | Path) -> Path:
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    return index_path


def load_index(index_path: str | Path) -> faiss.Index:
    return faiss.read_index(str(Path(index_path)))
