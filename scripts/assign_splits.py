"""Populate the ``split`` column of ``data/metadata/dataset.csv``.

Strategy: deterministic stride-based split inside each (label, video_stem)
group. By default every 5th frame becomes a ``query`` and the rest are
``gallery`` (≈ 80/20). The first frame of every group is forced to ``gallery``
so the index is never empty for any (floor, area).

Why per-video stride and not random?

* Adjacent frames at 1 FPS look almost identical, so a random shuffle would
  put near-duplicates in both gallery and query and the metrics would be
  meaningless.
* Stride is reproducible without seeds and produces the same split on every
  machine, which matters for replicating the evaluation results.

This is acknowledged to be an *easy* split because every query has a near
neighbour in the gallery (same video, frames N-2 / N+2). Top-K accuracy will
look good but does **not** prove generalisation across devices or lighting.
The README's failure-analysis section spells this out.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import DEFAULT_DATASET_CSV


GALLERY = "gallery"
QUERY = "query"

# Match "<stem>_frame_000123" -> capture stem (e.g. "f10_central_main") and frame
# number (123). Used for stable per-video ordering even if rows are shuffled.
FRAME_PATTERN = re.compile(r"^(?P<stem>.+)_frame_(?P<num>\d+)\.[A-Za-z0-9]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate the dataset.csv 'split' column deterministically.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_DATASET_CSV),
        help="Path to the dataset CSV.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=5,
        help=(
            "Stride for query frames. --every 5 -> 1 query every 5 frames "
            "(≈ 20%% query, 80%% gallery). Must be >= 2."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("fill_only", "overwrite"),
        default="fill_only",
        help=(
            "fill_only (default): only assign splits to rows that are still "
            "empty. overwrite: reassign every row."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip writing a .bak copy of the CSV before overwriting it.",
    )
    return parser.parse_args()


def parse_video_stem(image_path: str) -> tuple[str, int]:
    name = Path(image_path).name
    match = FRAME_PATTERN.match(name)
    if not match:
        # Fallback: treat whole stem as group, frame 0. Unmatched files are rare
        # but we don't want to crash here.
        return Path(image_path).stem, 0
    return match.group("stem"), int(match.group("num"))


def assign_split_for_group(group: pd.DataFrame, every: int) -> pd.Series:
    """Return a Series of split labels aligned with ``group`` rows.

    The frame at position 0 inside the group is always ``gallery`` so the
    gallery is never empty. Every ``every``-th subsequent frame becomes a
    ``query``.
    """
    ordered = group.sort_values("_frame_num")
    splits = []
    for position, _ in enumerate(ordered.itertuples(index=False)):
        if position == 0:
            splits.append(GALLERY)
        elif position % every == 0:
            splits.append(QUERY)
        else:
            splits.append(GALLERY)
    return pd.Series(splits, index=ordered.index)


def main() -> int:
    args = parse_args()
    if args.every < 2:
        print("[error] --every must be >= 2 (otherwise everything becomes query).", file=sys.stderr)
        return 1

    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        print(f"[error] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        df["split"] = ""

    # Normalise NaN to empty string for the "empty" check.
    df["split"] = df["split"].fillna("").astype(str)

    # Annotate helper columns for grouping.
    parsed = df["image_path"].astype(str).apply(parse_video_stem)
    df["_video_stem"] = [stem for stem, _ in parsed]
    df["_frame_num"] = [num for _, num in parsed]

    new_split_col = df["split"].copy()
    rows_processed = 0
    for (label, stem), group in df.groupby(["label", "_video_stem"], sort=False):
        if args.mode == "fill_only":
            empty_mask = group["split"].eq("")
            if not empty_mask.any():
                continue
            target_group = group
        else:
            target_group = group

        assignment = assign_split_for_group(target_group, args.every)
        if args.mode == "fill_only":
            empty_idx = target_group.index[target_group["split"].eq("")]
            new_split_col.loc[empty_idx] = assignment.loc[empty_idx]
            rows_processed += len(empty_idx)
        else:
            new_split_col.loc[target_group.index] = assignment
            rows_processed += len(target_group)

    df["split"] = new_split_col
    df = df.drop(columns=["_video_stem", "_frame_num"])

    if not args.no_backup:
        backup_path = csv_path.with_suffix(csv_path.suffix + ".bak")
        shutil.copy2(csv_path, backup_path)
        print(f"[backup] Wrote {backup_path}")

    df.to_csv(csv_path, index=False)

    counts = df["split"].value_counts(dropna=False).to_dict()
    print()
    print("=== Split summary ===")
    print(f"Mode: {args.mode} | rows updated: {rows_processed}")
    for split_name in (GALLERY, QUERY, ""):
        if split_name in counts:
            label = split_name or "(empty)"
            print(f"  {label}: {counts[split_name]}")
    print()
    by_floor = (
        df.assign(split=df["split"])
        .groupby(["label", "split"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    print("Per-floor breakdown:")
    print(by_floor.to_string())
    print()
    print(f"Saved CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
