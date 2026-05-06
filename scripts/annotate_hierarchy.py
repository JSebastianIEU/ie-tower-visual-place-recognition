"""Populate the ``area``, ``section`` and ``floor_range`` columns in
``data/metadata/dataset.csv`` so the hierarchical predictor (PR-A) has the
labels it needs.

What each column means:

* ``area``        — what kind of space the frame shows (hallway, elevator,
                    stairs, classroom, basement_lift, chill_lounge, …).
                    Extracted from the filename.
* ``section``     — coarse split: ``basement`` vs ``above-ground``.
* ``floor_range`` — basement / lowrise (floor 3-9) / midrise (floor 10-16) /
                    highrise (floor 17-23). Always equals ``basement`` for
                    basements so the column is non-null for every row.

The script is **idempotent**: rows whose three target columns are already
filled in are left untouched. Pass ``--force`` to re-derive everything.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the alias map from sync_drive_data so any new alias added there is
# picked up here automatically.
from scripts.sync_drive_data import AREA_ALIASES  # noqa: E402
from src.utils.config import DEFAULT_DATASET_CSV  # noqa: E402


# Sorted longest-first so longest-prefix matching wins (open_area before open).
_KNOWN_AREAS = sorted(AREA_ALIASES.keys(), key=len, reverse=True)


# Floor-range buckets for above-ground floors. Basements are handled
# separately because they bypass the f<NN>_ → floor<N> mapping.
_FLOOR_RANGES: list[tuple[range, str]] = [
    (range(3, 10), "lowrise"),    # floor3 .. floor9
    (range(10, 17), "midrise"),   # floor10 .. floor16
    (range(17, 24), "highrise"),  # floor17 .. floor23
]


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def derive_area(image_path: str) -> str:
    """Extract the area token from a frame filename.

    Above-ground frames look like ``f10_hallway_left_frame_000001.jpg``: the
    area is the first token (``hallway``) after the floor prefix; we then
    fold it through ``AREA_ALIASES`` so ``cafeteria`` becomes ``central``,
    ``class`` becomes ``classroom``, etc.

    Basement frames look like ``b0_basement_lift_c030_frame_000001.jpg``:
    the descriptor is the ``c<NNN>`` clip ID, so the area is everything
    between the floor prefix and the clip ID. Multi-word areas like
    ``chill_lounge`` are preserved.
    """
    stem = Path(image_path).stem  # e.g. "b3_chill_lounge_c029_frame_000007"
    match = re.match(
        r"^(?P<prefix>(?:f\d{2}|b\d))_(?P<body>.+)_frame_\d+$",
        stem,
        re.IGNORECASE,
    )
    if not match:
        return ""
    prefix = match.group("prefix").lower()
    body = match.group("body")

    if prefix.startswith("b"):
        # Basement: strip the trailing _c<NNN> clip-ID descriptor.
        clip_match = re.match(r"^(?P<area>.+)_c\d+$", body, re.IGNORECASE)
        if clip_match:
            return clip_match.group("area").lower()
        return body.lower()

    # Above-ground: longest known-area prefix wins, otherwise first token.
    body_lower = body.lower()
    for alias in _KNOWN_AREAS:
        if body_lower == alias or body_lower.startswith(alias + "_"):
            return AREA_ALIASES[alias]
    first_token = body_lower.split("_", 1)[0]
    return AREA_ALIASES.get(first_token, first_token)


def derive_section(label: str) -> str:
    label = str(label).lower()
    if label.startswith("basement"):
        return "basement"
    if label.startswith("floor"):
        return "above-ground"
    return ""


def derive_floor_range(label: str) -> str:
    label = str(label).lower()
    if label.startswith("basement"):
        return "basement"
    match = re.match(r"^floor(\d+)$", label)
    if not match:
        return ""
    floor_num = int(match.group(1))
    for bucket, name in _FLOOR_RANGES:
        if floor_num in bucket:
            return name
    return ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate the area / section / floor_range columns in "
            "data/metadata/dataset.csv. Idempotent — only fills rows that "
            "currently have empty values."
        )
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_DATASET_CSV),
        help="Path to dataset.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute every row, even if the three columns are already non-empty.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"[error] {csv_path} does not exist.", file=sys.stderr)
        return 1

    df = pd.read_csv(csv_path)

    for column in ("area", "section", "floor_range"):
        if column not in df.columns:
            df[column] = ""

    # Treat NaN as empty for the idempotency check.
    for column in ("area", "section", "floor_range"):
        df[column] = df[column].fillna("").astype(str)

    if args.force:
        rows_to_update = df.index
    else:
        empty_mask = (
            df["area"].eq("") | df["section"].eq("") | df["floor_range"].eq("")
        )
        rows_to_update = df.index[empty_mask]

    if len(rows_to_update) == 0:
        print("[annotate] nothing to do — every row already has area / section / floor_range.")
        return 0

    print(f"[annotate] populating {len(rows_to_update)}/{len(df)} rows")

    for idx in rows_to_update:
        row = df.loc[idx]
        df.at[idx, "area"] = derive_area(str(row["image_path"]))
        df.at[idx, "section"] = derive_section(str(row["label"]))
        df.at[idx, "floor_range"] = derive_floor_range(str(row["label"]))

    df.to_csv(csv_path, index=False)
    print(f"[annotate] wrote {csv_path}")
    print()
    print("Distinct values per new column:")
    for column in ("area", "section", "floor_range"):
        counts = df[column].value_counts(dropna=False)
        print(f"  {column}: {len(counts)} unique")
        for value, count in counts.head(15).items():
            shown = repr(value) if value == "" else value
            print(f"    {shown:<25s}  {count}")
        if len(counts) > 15:
            print(f"    ... +{len(counts) - 15} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
