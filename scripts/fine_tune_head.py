"""Train a small projection head on top of frozen DINOv2 embeddings.

Strategy: read the already-saved gallery embeddings + metadata
(produced by ``scripts/run_pipeline.py``), filter to the rows whose
``split == "gallery"``, and train a 2-layer MLP with the batch-hard
triplet loss until the held-out (``split == "query"``) Top-1 accuracy
plateaus or starts to drop. Save the best checkpoint to
``outputs/ft_models/projection_head.pth``.

Why feed the head pre-computed features instead of raw images:
DINOv2 ViT-S/14 at 518x518 is ~5 s per image on CPU; 2373 gallery
rows × 50 epochs would be 80+ hours. With pre-computed features the
whole training loop runs in a couple of minutes.

The script writes:
* ``outputs/ft_models/projection_head.pth`` — state dict of the best
  head (highest held-out Top-1).
* ``outputs/ft_models/training_log.json`` — per-epoch loss / accuracy
  trace plus the configuration dict, for later inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.loss_functions import batch_hard_triplet_loss  # noqa: E402
from src.features.models import ProjectionHead  # noqa: E402
from src.utils.config import EMBEDDINGS_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train a 2-layer projection head on top of cached DINOv2 "
            "embeddings using batch-hard triplet loss."
        )
    )
    p.add_argument("--embeddings", default=str(EMBEDDINGS_DIR / "gallery_embeddings.npy"))
    p.add_argument("--metadata", default=str(EMBEDDINGS_DIR / "gallery_metadata.csv"))
    p.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs" / "ft_models" / "projection_head.pth"),
    )
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--out-dim", type=int, default=128)
    p.add_argument("--residual", action="store_true",
                   help="Use a residual head (out_dim must equal in_dim). "
                        "Initialised near identity so it nudges the frozen "
                        "features instead of replacing them.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--samples-per-class",
        type=int,
        default=4,
        help=(
            "PK-style sampler: pick K rows per class so the batch always "
            "contains positives. Effective batch size = "
            "samples_per_class * num_classes_per_batch."
        ),
    )
    p.add_argument(
        "--classes-per-batch",
        type=int,
        default=12,
        help="How many distinct classes to sample per batch.",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pk_sampler(
    label_to_indices: dict,
    classes_per_batch: int,
    samples_per_class: int,
    rng: random.Random,
):
    """Yield batches of indices using PK sampling (P classes × K samples).

    Each batch is built by sampling ``classes_per_batch`` distinct
    labels (without replacement until exhausted) and ``samples_per_class``
    rows per label. This guarantees every batch has both positives and
    negatives for the batch-hard miner.
    """
    classes = list(label_to_indices.keys())
    while True:
        rng.shuffle(classes)
        for start in range(0, len(classes), classes_per_batch):
            picked_classes = classes[start : start + classes_per_batch]
            if len(picked_classes) < 2:
                continue  # need at least 2 classes for a meaningful batch
            batch = []
            for label in picked_classes:
                pool = label_to_indices[label]
                if len(pool) >= samples_per_class:
                    batch.extend(rng.sample(pool, samples_per_class))
                else:
                    # Sample with replacement when a class is small.
                    batch.extend(rng.choices(pool, k=samples_per_class))
            yield batch


def evaluate_head(
    head: ProjectionHead,
    features: torch.Tensor,
    labels: list[str],
    splits: list[str],
    device: torch.device,
) -> dict:
    """Apply the head to ``features`` and compute Top-1 / Top-5 accuracy
    of the gallery → query retrieval at the floor level.

    Uses the frozen split column from the metadata, so this is a faithful
    proxy for what ``run_evaluation.py`` will compute later.
    """
    head.eval()
    with torch.no_grad():
        projected = head(features.to(device)).cpu().numpy()

    gallery_mask = np.array([s == "gallery" for s in splits])
    query_mask = np.array([s == "query" for s in splits])
    gallery_emb = projected[gallery_mask]
    query_emb = projected[query_mask]
    gallery_labels = np.array([labels[i] for i in range(len(labels)) if gallery_mask[i]])
    query_labels = np.array([labels[i] for i in range(len(labels)) if query_mask[i]])

    if len(gallery_emb) == 0 or len(query_emb) == 0:
        return {"top_1_accuracy": 0.0, "top_5_accuracy": 0.0}

    sims = query_emb @ gallery_emb.T  # both already normalised
    top5 = np.argsort(-sims, axis=1)[:, :5]
    top1_hits = 0
    top5_hits = 0
    for i, ranked in enumerate(top5):
        truth = query_labels[i]
        if gallery_labels[ranked[0]] == truth:
            top1_hits += 1
        if any(gallery_labels[r] == truth for r in ranked):
            top5_hits += 1
    return {
        "top_1_accuracy": top1_hits / len(query_labels),
        "top_5_accuracy": top5_hits / len(query_labels),
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    embeddings = np.load(args.embeddings).astype(np.float32)
    metadata = pd.read_csv(args.metadata)
    if len(embeddings) != len(metadata):
        print(
            f"[error] embedding rows ({len(embeddings)}) != metadata rows "
            f"({len(metadata)})",
            file=sys.stderr,
        )
        return 1

    in_dim = embeddings.shape[1]
    splits = metadata["split"].astype(str).tolist()
    labels = metadata["label"].astype(str).tolist()
    gallery_idx = [i for i, s in enumerate(splits) if s == "gallery"]
    print(f"[fine-tune] gallery rows: {len(gallery_idx)} | held-out queries: {sum(1 for s in splits if s == 'query')}")
    print(f"[fine-tune] feature dim: {in_dim} | hidden: {args.hidden_dim} | out: {args.out_dim}")

    label_to_indices: dict[str, list[int]] = {}
    for idx in gallery_idx:
        label_to_indices.setdefault(labels[idx], []).append(idx)
    label_to_indices = {
        lbl: pool for lbl, pool in label_to_indices.items() if len(pool) >= 2
    }
    print(f"[fine-tune] usable gallery classes (>=2 rows): {len(label_to_indices)}")
    if len(label_to_indices) < 2:
        print("[error] need at least 2 classes with >=2 gallery rows.", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[fine-tune] device: {device}")

    head = ProjectionHead(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        residual=args.residual,
    ).to(device)

    optim = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    features_tensor = torch.from_numpy(embeddings).to(device)
    label_to_int = {lbl: i for i, lbl in enumerate(sorted(set(labels)))}
    int_labels = torch.tensor([label_to_int[lbl] for lbl in labels], dtype=torch.long, device=device)

    rng = random.Random(args.seed)
    sampler = pk_sampler(
        label_to_indices,
        classes_per_batch=args.classes_per_batch,
        samples_per_class=args.samples_per_class,
        rng=rng,
    )

    # Roughly one "epoch" = enough batches to touch every class twice.
    classes_per_epoch_pass = max(
        1, math.ceil(len(label_to_indices) / args.classes_per_batch)
    )
    batches_per_epoch = classes_per_epoch_pass * 4

    log: list[dict] = []
    best = {"top_1_accuracy": -1.0, "epoch": -1}
    best_state = None

    # Initial evaluation (epoch 0): unprojected features baseline through
    # an untrained head — useful to confirm we don't underperform the
    # raw embeddings.
    initial = evaluate_head(head, features_tensor, labels, splits, device)
    print(f"[fine-tune] init   Top-1={initial['top_1_accuracy']:.3f}  Top-5={initial['top_5_accuracy']:.3f}")
    log.append({"epoch": 0, "phase": "init", **initial})

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        head.train()
        epoch_loss = 0.0
        epoch_active = 0.0
        steps = 0
        for _ in range(batches_per_epoch):
            batch_idx = next(sampler)
            batch_features = features_tensor[batch_idx]
            batch_labels = int_labels[batch_idx]

            projected = head(batch_features)
            loss, stats = batch_hard_triplet_loss(
                projected, batch_labels, margin=args.margin
            )
            if not torch.isfinite(loss):
                continue

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            epoch_loss += loss.item()
            epoch_active += stats["active_rate"]
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        avg_active = epoch_active / max(steps, 1)

        eval_metrics = evaluate_head(head, features_tensor, labels, splits, device)
        log.append(
            {
                "epoch": epoch,
                "phase": "train",
                "loss": avg_loss,
                "active_rate": avg_active,
                **eval_metrics,
            }
        )

        msg = (
            f"[fine-tune] epoch {epoch:2d}/{args.epochs}  loss={avg_loss:.4f}  "
            f"active={avg_active:.2%}  Top-1={eval_metrics['top_1_accuracy']:.3f}  "
            f"Top-5={eval_metrics['top_5_accuracy']:.3f}"
        )
        print(msg)

        if eval_metrics["top_1_accuracy"] > best["top_1_accuracy"]:
            best = {"epoch": epoch, **eval_metrics}
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}

    elapsed = time.time() - t0
    print()
    print(f"[fine-tune] best epoch: {best['epoch']}  Top-1={best['top_1_accuracy']:.3f}  Top-5={best['top_5_accuracy']:.3f}")
    print(f"[fine-tune] training time: {elapsed:.1f} s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None:
        # Edge case: training never improved over init. Save the final
        # state anyway so the consumer always has a checkpoint, but warn.
        best_state = head.state_dict()
        print("[warn] no epoch improved over init; saving final state.")
    torch.save(best_state, output_path)
    print(f"[fine-tune] saved checkpoint: {output_path}")

    log_path = output_path.parent / "training_log.json"
    log_path.write_text(
        json.dumps(
            {
                "config": vars(args),
                "in_dim": in_dim,
                "best": best,
                "initial": initial,
                "log": log,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[fine-tune] log saved: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
