"""Evaluation metrics for the four-tier hierarchical predictor.

Three families of numbers:

1. ``top_k_<level>_accuracy`` — how often the ground-truth value of a
   given hierarchy column appears in the Top-K matches at that level.
   Reported for ``floor`` (the existing Top-K), ``floor_range``,
   ``section`` and ``area``.

2. ``tier_resolution_rate`` — what fraction of held-out queries are
   resolved at each tier with the configured confidence threshold.
   Higher numbers at the ``floor`` tier are good; high numbers at the
   ``area`` tier mean the model is "stepping down" a lot.

3. ``tier_when_resolved_accuracy`` — when the predictor decides to
   answer at a given tier, how often is its answer correct? This is
   the metric that matters for the user experience: an answer at the
   ``area`` tier is worse than ``floor`` only if the model is no longer
   above-chance at the area tier.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.retrieval.build_index import build_faiss_index
from src.retrieval.hierarchical import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    hierarchical_predict,
)
from src.retrieval.search import search_index


def _column_top_k_accuracy(
    retrieved_per_query: list[list[Optional[str]]],
    truth_per_query: list[Optional[str]],
    top_ks: Iterable[int],
) -> dict[int, float]:
    """Top-K accuracy where 'correct' means the ground-truth value of a
    hierarchy column appears in the top-K matches' values for that column.

    Queries whose truth is missing (None / empty) are ignored.
    """
    out: dict[int, float] = {}
    for k in top_ks:
        hits = 0
        considered = 0
        for retrieved, truth in zip(retrieved_per_query, truth_per_query):
            if truth is None or truth == "":
                continue
            considered += 1
            if any(v == truth for v in retrieved[:k] if v):
                hits += 1
        out[k] = (hits / considered) if considered else 0.0
    return out


def evaluate_hierarchical(
    gallery_embeddings: np.ndarray,
    gallery_metadata: pd.DataFrame,
    query_embeddings: np.ndarray,
    query_metadata: pd.DataFrame,
    *,
    top_ks: Iterable[int] = (1, 5),
    metric: str = "cosine",
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Compute hierarchical metrics on a held-out query set.

    Builds a fresh FAISS index from ``gallery_embeddings`` (so the
    function is self-contained and can be called in isolation), runs the
    queries through it, and aggregates per-level accuracy plus the tier
    resolution rate.
    """
    gallery_metadata = gallery_metadata.reset_index(drop=True)
    query_metadata = query_metadata.reset_index(drop=True)

    top_ks = tuple(sorted(set(top_ks)))
    if not top_ks:
        raise ValueError("top_ks must contain at least one value.")

    max_k = max(top_ks)
    index = build_faiss_index(gallery_embeddings, metric=metric)
    ranked = search_index(
        query_embeddings=query_embeddings,
        index=index,
        metadata=gallery_metadata,
        top_k=max_k,
        metric=metric,
    )

    metrics: dict[str, dict] = {}

    for column, level in (
        ("label", "floor"),
        ("floor_range", "floor_range"),
        ("section", "section"),
        ("area", "area"),
    ):
        if column not in query_metadata.columns:
            continue
        truth = [
            (str(v).strip() if pd.notna(v) and str(v).strip() else None)
            for v in query_metadata[column]
        ]
        retrieved_per_query = []
        for results in ranked:
            retrieved_per_query.append(
                [getattr(r, column if column != "label" else "label") for r in results]
                if column == "label"
                else [getattr(r, column) for r in results]
            )
        per_k = _column_top_k_accuracy(retrieved_per_query, truth, top_ks)
        for k, value in per_k.items():
            metrics[f"top_{k}_{level}_accuracy"] = value

    # Tier resolution + per-tier when-resolved accuracy.
    tier_counts = {"floor": 0, "floor_range": 0, "section": 0, "area": 0}
    tier_correct = {"floor": 0, "floor_range": 0, "section": 0, "area": 0}
    total_resolved = 0
    for results, (_, query_row) in zip(ranked, query_metadata.iterrows()):
        prediction = hierarchical_predict(results, threshold=threshold)
        best = prediction.get("best") or {}
        level = prediction.get("selected_level")
        if level not in tier_counts:
            continue
        tier_counts[level] += 1
        total_resolved += 1
        truth_field = "label" if level == "floor" else level
        if truth_field not in query_metadata.columns:
            continue
        truth_value = query_row.get(truth_field)
        truth_value = (
            str(truth_value).strip() if pd.notna(truth_value) else ""
        )
        if truth_value and best.get("label") == truth_value:
            tier_correct[level] += 1

    metrics["tier_resolution_rate"] = {
        level: (tier_counts[level] / total_resolved if total_resolved else 0.0)
        for level in tier_counts
    }
    metrics["tier_when_resolved_accuracy"] = {
        level: (tier_correct[level] / tier_counts[level] if tier_counts[level] else 0.0)
        for level in tier_counts
    }
    metrics["confidence_threshold"] = threshold
    metrics["num_queries"] = len(query_metadata)
    return metrics
