from typing import Iterable

import numpy as np
import pandas as pd

from src.evaluation.hierarchical_metrics import evaluate_hierarchical
from src.evaluation.metrics import mean_average_precision, top_k_accuracy
from src.retrieval.build_index import build_faiss_index
from src.retrieval.hierarchical import DEFAULT_CONFIDENCE_THRESHOLD
from src.retrieval.search import search_index


def evaluate_retrieval(
    gallery_embeddings: np.ndarray,
    gallery_metadata: pd.DataFrame,
    query_embeddings: np.ndarray | None = None,
    query_metadata: pd.DataFrame | None = None,
    top_ks: Iterable[int] = (1, 5),
    metric: str = "cosine",
    hierarchical_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
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

    # ---- Per-class breakdown + confusion analysis ------------------------
    # These help diagnose which labels the model is good at and which ones
    # it confuses, without changing the headline metrics above.
    per_class: dict[str, dict] = {}
    confusion: dict[str, dict[str, int]] = {}
    for gt, ranked in zip(ground_truth_labels, retrieved_labels):
        bucket = per_class.setdefault(
            gt,
            {"queries": 0, "top1_correct": 0, "top5_correct": 0},
        )
        bucket["queries"] += 1
        if ranked and ranked[0] == gt:
            bucket["top1_correct"] += 1
        if gt in ranked[:5]:
            bucket["top5_correct"] += 1
        if ranked and ranked[0] != gt:
            row = confusion.setdefault(gt, {})
            row[ranked[0]] = row.get(ranked[0], 0) + 1

    for bucket in per_class.values():
        n = bucket["queries"]
        bucket["top1_accuracy"] = bucket["top1_correct"] / n if n else 0.0
        bucket["top5_accuracy"] = bucket["top5_correct"] / n if n else 0.0

    metrics["per_class"] = per_class
    metrics["confusion_top_misses"] = confusion
    metrics["num_queries"] = len(ground_truth_labels)

    # ---- Hierarchical / 4-tier metrics -----------------------------------
    # Only meaningful when the metadata exposes the hierarchy columns
    # produced by scripts/annotate_hierarchy.py. Skipped for legacy CSVs
    # so the existing top-K numbers are unchanged for old runs.
    has_hierarchy = all(
        column in query_metadata.columns
        for column in ("floor_range", "section", "area")
    )
    if has_hierarchy:
        metrics["hierarchical"] = evaluate_hierarchical(
            gallery_embeddings=gallery_embeddings,
            gallery_metadata=gallery_metadata,
            query_embeddings=query_embeddings,
            query_metadata=query_metadata,
            top_ks=top_ks,
            metric=metric,
            threshold=hierarchical_threshold,
        )
    return metrics
