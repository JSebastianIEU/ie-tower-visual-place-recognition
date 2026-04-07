from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.retrieval.faiss_utils import ensure_float32, l2_normalize


@dataclass
class SearchResult:
    rank: int
    index: int
    score: float
    image_path: str
    label: str


def search_index(
    query_embeddings: np.ndarray,
    index,
    metadata: pd.DataFrame,
    top_k: int = 5,
    metric: str = "cosine",
) -> list[list[SearchResult]]:
    metadata = metadata.reset_index(drop=True)
    queries = np.atleast_2d(ensure_float32(query_embeddings))

    if metric.lower() == "cosine":
        queries = l2_normalize(queries)

    scores, indices = index.search(queries, top_k)
    all_results: list[list[SearchResult]] = []

    for query_scores, query_indices in zip(scores, indices):
        query_results = []
        for rank, (score, neighbor_index) in enumerate(
            zip(query_scores, query_indices), start=1
        ):
            if neighbor_index < 0:
                continue

            row = metadata.iloc[int(neighbor_index)]
            query_results.append(
                SearchResult(
                    rank=rank,
                    index=int(neighbor_index),
                    score=float(score),
                    image_path=str(row.get("image_path", "")),
                    label=str(row.get("label", "")),
                )
            )

        all_results.append(query_results)

    return all_results
