"""Hierarchical / progressive-specificity predictor.

The retrieval pipeline returns the Top-K gallery matches. From those we
compute a vote at four levels of specificity:

    1. ``floor``        — the most specific label the dataset exposes
                          (``floor10``, ``basement3``, …).
    2. ``floor_range``  — coarser bucket (``lowrise`` / ``midrise`` /
                          ``highrise`` / ``basement``).
    3. ``section``      — coarsest spatial bucket (``above-ground`` /
                          ``basement``).
    4. ``area``         — orthogonal: the type of space (``hallway``,
                          ``elevator``, ``stairs``, ``classroom``,
                          ``chill_lounge``, …). Useful as the last-ditch
                          fallback when the model can't pin down vertical
                          location at all.

For every level we count how many of the Top-K gallery matches agree on a
single value and divide by ``K`` to get a confidence ratio. The selected
"best" tier is the most-specific one whose ratio meets a threshold
(default 0.6, i.e. 3 out of 5). The predictor always returns the full
fallback chain so the UI can show what the model would have said at each
tier.
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


# Specificity order, most specific first. The selected best is the first
# tier whose ratio meets the threshold; if none do, we still return the
# best available tier (the last entry, i.e. ``area``) so the user always
# sees *something*.
_TIER_ORDER: tuple[str, ...] = ("floor", "floor_range", "section", "area")


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
    """Compute the four-tier prediction from a Top-K result list.

    Returns a dict shaped like::

        {
          "best": {"level": "floor", "label": "floor12", "votes": 4, "k": 5,
                   "ratio": 0.8, "human_level": "exact floor"},
          "selected_level": "floor",
          "fallback_chain": [
            {"level": "floor",       "label": "floor12",      "votes": 4, ...},
            {"level": "floor_range", "label": "midrise",      "votes": 5, ...},
            {"level": "section",     "label": "above-ground", "votes": 5, ...},
            {"level": "area",        "label": "hallway",      "votes": 4, ...},
          ],
          "threshold": 0.6,
        }

    ``best`` is always populated. If no tier reaches the threshold, the
    least-specific tier (``area``) is selected so the consumer always has
    at least a coarse answer to show.
    """
    k = len(results)

    # Floor / floor_range / section come from the gallery row's columns.
    # ``label`` doubles as the floor level (the metadata's primary label).
    floor_vote_label, floor_votes, _ = _vote([r.label for r in results])
    range_vote_label, range_votes, _ = _vote([r.floor_range for r in results])
    section_vote_label, section_votes, _ = _vote([r.section for r in results])
    area_vote_label, area_votes, _ = _vote([r.area for r in results])

    tiers = {
        "floor":       TierVote("floor",       floor_vote_label,   floor_votes,   k),
        "floor_range": TierVote("floor_range", range_vote_label,   range_votes,   k),
        "section":     TierVote("section",     section_vote_label, section_votes, k),
        "area":        TierVote("area",        area_vote_label,    area_votes,    k),
    }

    fallback_chain = [tiers[level].as_dict() for level in _TIER_ORDER]

    selected_level = None
    for level in _TIER_ORDER:
        tier = tiers[level]
        if tier.label is None:
            continue
        if tier.ratio >= threshold:
            selected_level = level
            break

    if selected_level is None:
        # No tier hit the threshold — fall back to the least specific one
        # that has a label at all.
        for level in reversed(_TIER_ORDER):
            if tiers[level].label is not None:
                selected_level = level
                break

    best = tiers[selected_level].as_dict() if selected_level else None
    return {
        "best": best,
        "selected_level": selected_level,
        "fallback_chain": fallback_chain,
        "threshold": threshold,
    }


def explain_prediction(prediction: dict) -> str:
    """Render a one-line human-readable explanation of the chosen tier.

    Useful in the demo banner and in the Jupyter notebook output.
    """
    best = prediction.get("best")
    if not best or best.get("label") is None:
        return "No confident prediction at any tier."
    level = best["level"]
    label = best["label"]
    votes = best["votes"]
    k = best["k"]
    ratio = best["ratio"]

    if level == "floor":
        return f"Floor: {label} ({votes}/{k} = {ratio:.0%}, exact)."
    if level == "floor_range":
        return (
            f"Range: {label} ({votes}/{k} = {ratio:.0%}). "
            "Top-K split between same-range floors — exact floor uncertain."
        )
    if level == "section":
        return (
            f"Section: {label} ({votes}/{k} = {ratio:.0%}). "
            "Model confident only about above-ground vs basement."
        )
    if level == "area":
        return (
            f"Area type: {label} ({votes}/{k} = {ratio:.0%}). "
            "Vertical location is unclear — most matches share this kind of space."
        )
    return f"{level}: {label} ({votes}/{k} = {ratio:.0%})."
