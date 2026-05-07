"""Triplet-loss objectives for fine-tuning a projection head on top of
the frozen DINOv2 backbone.

Two flavours are exposed:

* ``triplet_margin_loss(anchor, positive, negative, margin)`` —
  classical margin loss for explicit triplets supplied by the caller.
  Useful for unit tests and for offline triplet mining.
* ``batch_hard_triplet_loss(features, labels, margin)`` — online
  batch-hard mining: for every anchor in the batch we use the *furthest*
  same-label point as the positive and the *closest* different-label
  point as the negative. This is the standard recipe from "In Defense
  of the Triplet Loss for Person Re-Identification" (Hermans et al.,
  2017) and tends to outperform random triplets by a healthy margin.

Distance is cosine-distance (``1 - cos_sim``) which assumes the inputs
are L2-normalised. The training loop is responsible for normalising the
projection-head output before calling this function.
"""

from __future__ import annotations

import torch


def _pairwise_cosine_distance(x: torch.Tensor) -> torch.Tensor:
    """Return the (N, N) cosine-distance matrix for L2-normalised ``x``."""
    sim = x @ x.t()
    # Numerical noise can push values slightly outside [-1, 1].
    sim = sim.clamp(-1.0, 1.0)
    return 1.0 - sim


def triplet_margin_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Classical triplet margin loss with cosine distance.

    ``anchor`` / ``positive`` / ``negative`` are each ``(batch, dim)``
    tensors of L2-normalised vectors.
    """
    pos_dist = 1.0 - (anchor * positive).sum(dim=1).clamp(-1.0, 1.0)
    neg_dist = 1.0 - (anchor * negative).sum(dim=1).clamp(-1.0, 1.0)
    return torch.clamp(pos_dist - neg_dist + margin, min=0.0).mean()


def batch_hard_triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """Online batch-hard triplet mining loss.

    Parameters
    ----------
    features : (N, D) L2-normalised tensor.
    labels   : (N,) integer or long tensor with one label per row.
    margin   : positive scalar; the minimum gap between hardest positive
               and hardest negative distances we want.

    Returns
    -------
    loss     : scalar tensor (mean across rows that have both a valid
               positive and a valid negative in the batch).
    stats    : dict with mean hardest-positive / hardest-negative
               distances and the active-triplet rate (fraction of rows
               whose loss is non-zero) — useful to log per epoch and
               detect collapse.
    """
    if features.dim() != 2:
        raise ValueError("features must be (N, D)")
    n = features.size(0)

    distance = _pairwise_cosine_distance(features)
    label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
    eye = torch.eye(n, dtype=torch.bool, device=features.device)

    # Hardest positive: max distance among same-label, excluding self.
    pos_mask = label_eq & ~eye
    pos_distance = distance.masked_fill(~pos_mask, float("-inf"))
    hardest_pos, _ = pos_distance.max(dim=1)
    has_positive = pos_mask.any(dim=1)

    # Hardest negative: min distance among different-label.
    neg_mask = ~label_eq
    neg_distance = distance.masked_fill(~neg_mask, float("inf"))
    hardest_neg, _ = neg_distance.min(dim=1)
    has_negative = neg_mask.any(dim=1)

    valid = has_positive & has_negative
    if not valid.any():
        zero = features.new_zeros(())
        return zero, {
            "hardest_pos_mean": float("nan"),
            "hardest_neg_mean": float("nan"),
            "active_rate": 0.0,
            "valid_rows": 0,
        }

    pos_d = hardest_pos[valid]
    neg_d = hardest_neg[valid]
    raw_loss = pos_d - neg_d + margin
    loss = raw_loss.clamp(min=0.0)

    stats = {
        "hardest_pos_mean": float(pos_d.mean().item()),
        "hardest_neg_mean": float(neg_d.mean().item()),
        "active_rate": float((raw_loss > 0).float().mean().item()),
        "valid_rows": int(valid.sum().item()),
    }
    return loss.mean(), stats
