from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.metrics import mean_average_precision, top_k_accuracy
from src.retrieval.build_index import build_faiss_index
from src.retrieval.search import search_index


def evaluate_retrieval(
    gallery_embeddings: np.ndarray,
    gallery_metadata: pd.DataFrame,
    query_embeddings: np.ndarray | None = None,
    query_metadata: pd.DataFrame | None = None,
    top_ks: Iterable[int] = (1, 5),
    metric: str = "cosine",
) -> dict:
    gallery_metadata = gallery_metadata.reset_index(drop=True)
    same_gallery = query_embeddings is None and query_metadata is None

    if (query_embeddings is None) != (query_metadata is None):
        raise ValueError(
            "query_embeddings and query_metadata must be provided together."
        )

    if same_gallery:
        query_embeddings = gallery_embeddings
        query_metadata = gallery_metadata.copy()
    else:
        query_metadata = query_metadata.reset_index(drop=True)

    top_ks = tuple(sorted(set(top_ks)))
    if not top_ks:
        raise ValueError("top_ks must contain at least one value.")

    index = build_faiss_index(gallery_embeddings, metric=metric)
    search_depth = max(top_ks) + (1 if same_gallery else 0)
    ranked_results = search_index(
        query_embeddings=query_embeddings,
        index=index,
        metadata=gallery_metadata,
        top_k=search_depth,
        metric=metric,
    )

    ground_truth_labels = query_metadata["label"].astype(str).tolist()
    query_paths = (
        query_metadata["image_path"].astype(str).tolist()
        if "image_path" in query_metadata.columns
        else [None] * len(query_metadata)
    )

    retrieved_labels = []
    relevance_lists = []

    for query_path, ground_truth, current_results in zip(
        query_paths, ground_truth_labels, ranked_results
    ):
        filtered_results = []
        for result in current_results:
            if same_gallery and query_path is not None and result.image_path == query_path:
                continue

            filtered_results.append(result)
            if len(filtered_results) == max(top_ks):
                break

        labels = [result.label for result in filtered_results]
        retrieved_labels.append(labels)
        relevance_lists.append([1 if label == ground_truth else 0 for label in labels])

    metrics = {
        f"top_{k}_accuracy": top_k_accuracy(retrieved_labels, ground_truth_labels, k)
        for k in top_ks
    }
    metrics["mAP"] = mean_average_precision(relevance_lists)
    return metrics
