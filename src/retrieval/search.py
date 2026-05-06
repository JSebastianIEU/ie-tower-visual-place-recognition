from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.retrieval.faiss_utils import ensure_float32, l2_normalize


@dataclass
class SearchResult:
    """One retrieved gallery row.

    The four hierarchy fields (``area``, ``section``, ``floor_range``) are
    populated from the matching columns in the gallery metadata when those
    columns exist. They are ``None`` for legacy CSVs that have not been
    passed through ``scripts/annotate_hierarchy.py`` yet, so consumers can
    feature-detect with ``if result.area is not None:`` instead of
    branching on the metadata schema.
    """

    rank: int
    index: int
    score: float
    image_path: str
    label: str
    area: Optional[str] = None
    section: Optional[str] = None
    floor_range: Optional[str] = None


def _column_value(row: pd.Series, column: str) -> Optional[str]:
    if column not in row.index:
        return None
    raw = row[column]
    if pd.isna(raw):
        return None
    text = str(raw).strip()
    return text or None


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
                    area=_column_value(row, "area"),
                    section=_column_value(row, "section"),
                    floor_range=_column_value(row, "floor_range"),
                )
            )

        all_results.append(query_results)

    return all_results
