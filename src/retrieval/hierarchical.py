"""Hierarchical / progressive-specificity predictor.

The retrieval pipeline returns the Top-K gallery matches. From those we
compute two complementary signals:

1. A **spatial fallback chain** (floor → floor_range → section) that
   answers "where are you in the building, vertically?". The predictor
   walks the chain from most specific to least specific and picks the
   first level whose Top-K vote ratio meets the configured confidence
   threshold. The full chain is always returned so the UI can show what
   the model would have answered at every tier.

2. An **orthogonal area tag** (hallway, elevator, stairs, classroom,
   chill_lounge, …) that answers "what kind of space is this?". This is
   not a step in the spatial fallback chain — it is shown alongside the
   selected tier as a secondary label. This is what changed between the
   first version of the predictor and the OCR PR: previously ``area``
   sat at the bottom of the chain and was never selected because
   ``section`` (only two values) virtually always met the threshold
   first. Splitting it out makes both signals legible at the same time.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from src.retrieval.search import SearchResult


# Default confidence threshold for promoting a tier from "fallback" to
# "best". Tuned against the held-out queries — see
# outputs/results/evaluation.json.tier_resolution_rate after running
# scripts/run_evaluation.py.
DEFAULT_CONFIDENCE_THRESHOLD = 0.6


# Spatial fallback chain — most specific first. The selected best is the
# first tier whose ratio meets the threshold; if none do, we still return
# the least-specific tier so the user always sees *something*.
_SPATIAL_TIER_ORDER: tuple[str, ...] = ("floor", "floor_range", "section")


# Human-readable label per tier — only used for explanations / UI banners.
_TIER_LABEL: dict[str, str] = {
    "floor":       "exact floor",
    "floor_range": "floor range",
    "section":     "section",
    "area":        "area type",
}


@dataclass
class TierVote:
    """A vote summary at one specificity level."""

    level: str
    label: Optional[str]
    votes: int
    k: int

    @property
    def ratio(self) -> float:
        return (self.votes / self.k) if self.k else 0.0

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "label": self.label,
            "votes": self.votes,
            "k": self.k,
            "ratio": self.ratio,
            "human_level": _TIER_LABEL.get(self.level, self.level),
        }


def _vote(values: list[Optional[str]]) -> tuple[Optional[str], int, int]:
    """Return (top_label, top_votes, total_non_null) for a list of labels."""
    cleaned = [v for v in values if v is not None and str(v).strip()]
    total = len(cleaned)
    if total == 0:
        return None, 0, 0
    counter = Counter(cleaned)
    label, votes = counter.most_common(1)[0]
    return label, votes, total


def hierarchical_predict(
    results: list[SearchResult],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Compute the spatial fallback chain plus the orthogonal area tag.

    Returns a dict shaped like::

        {
          "best": {"level": "floor", "label": "floor12", ...},
          "selected_level": "floor",
          "fallback_chain": [
            {"level": "floor",       "label": "floor12",      ...},
            {"level": "floor_range", "label": "midrise",      ...},
            {"level": "section",     "label": "above-ground", ...},
          ],
          "area_tag": {"level": "area", "label": "hallway", "votes": 4, ...},
          "threshold": 0.6,
        }

    ``best`` is always populated. If no spatial tier reaches the threshold,
    the least-specific spatial tier (``section``) is selected so the
    consumer always has at least a coarse spatial answer to show.
    ``area_tag`` is independent — it is the area-vote summary and gets
    rendered alongside whichever spatial tier was selected.
    """
    k = len(results)

    floor_label, floor_votes, _ = _vote([r.label for r in results])
    range_label, range_votes, _ = _vote([r.floor_range for r in results])
    section_label, section_votes, _ = _vote([r.section for r in results])
    area_label, area_votes, _ = _vote([r.area for r in results])

    spatial_tiers = {
        "floor":       TierVote("floor",       floor_label,   floor_votes,   k),
        "floor_range": TierVote("floor_range", range_label,   range_votes,   k),
        "section":     TierVote("section",     section_label, section_votes, k),
    }
    area_tier = TierVote("area", area_label, area_votes, k)

    fallback_chain = [spatial_tiers[level].as_dict() for level in _SPATIAL_TIER_ORDER]

    selected_level = None
    for level in _SPATIAL_TIER_ORDER:
        tier = spatial_tiers[level]
        if tier.label is None:
            continue
        if tier.ratio >= threshold:
            selected_level = level
            break

    if selected_level is None:
        # Nothing reached the threshold — fall back to the least specific
        # spatial tier that has a label at all.
        for level in reversed(_SPATIAL_TIER_ORDER):
            if spatial_tiers[level].label is not None:
                selected_level = level
                break

    best = spatial_tiers[selected_level].as_dict() if selected_level else None
    return {
        "best": best,
        "selected_level": selected_level,
        "fallback_chain": fallback_chain,
        "area_tag": area_tier.as_dict(),
        "threshold": threshold,
    }


def explain_prediction(prediction: dict) -> str:
    """Render a one-line human-readable explanation of the chosen tier
    plus the orthogonal area tag.
    """
    best = prediction.get("best")
    area_tag = prediction.get("area_tag") or {}
    area_label = area_tag.get("label")
    area_ratio = area_tag.get("ratio", 0.0)

    if not best or best.get("label") is None:
        return "No confident prediction at any spatial tier."

    level = best["level"]
    label = best["label"]
    votes = best["votes"]
    k = best["k"]
    ratio = best["ratio"]

    if level == "floor":
        head = f"Floor: {label} ({votes}/{k} = {ratio:.0%}, exact)"
    elif level == "floor_range":
        head = (
            f"Range: {label} ({votes}/{k} = {ratio:.0%}). "
            "Top-K split between same-range floors — exact floor uncertain."
        )
    else:  # section
        head = (
            f"Section: {label} ({votes}/{k} = {ratio:.0%}). "
            "Model confident only about above-ground vs basement."
        )

    if area_label:
        return f"{head}  ·  Looks like a {area_label} ({area_ratio:.0%} of matches)."
    return head + "."
