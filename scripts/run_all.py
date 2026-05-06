"""End-to-end orchestrator for the VPR pipeline.

Runs every stage in order and **auto-skips** stages whose output is already on
disk and up to date. Each individual stage script is idempotent on its own,
but this top-level wrapper saves the user from having to remember the order
or which steps still need running.

Stages:
  1. sync_drive_data.py            — pull team videos from Drive into data/raw_videos/
  2. extract_frames_and_update_csv.py — extract 1 FPS frames + update dataset.csv
  3. assign_splits.py              — populate the gallery / query split column
  4. run_pipeline.py               — extract embeddings + build FAISS index
  5. run_evaluation.py             — compute Top-K accuracy + mAP

Skip detection (per stage):
  1. SYNC      — skipped if data/raw_videos/ already contains at least
                 ``min_videos`` files matching the canonical regex.
  2. EXTRACT   — skipped if every CSV row's image_path resolves to an
                 existing JPG **and** every video in raw_videos/ has frames
                 on disk.
  3. SPLITS    — skipped if the CSV already has a non-empty ``split``
                 column for every row.
  4. PIPELINE  — skipped if outputs/embeddings/gallery_embeddings.npy and
                 outputs/index/gallery.index both exist.
  5. EVAL      — skipped if outputs/results/evaluation.json exists.

Force a re-run with ``--force`` (re-runs everything) or per-stage flags
``--force-sync``, ``--force-extract``, ``--force-splits``,
``--force-pipeline``, ``--force-eval``.

Skip a stage entirely with ``--skip-sync`` (useful offline) or the matching
``--skip-*`` flags.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    DEFAULT_DATASET_CSV,
    EMBEDDINGS_DIR,
    INDEX_DIR,
    PROCESSED_FRAMES_DIR,
    RAW_VIDEOS_DIR,
    RESULTS_DIR,
)


CANONICAL_VIDEO_REGEX = re.compile(
    r"^(?:f\d{2}|b\d)_[a-z0-9]+(?:_[a-z0-9]+)+\.(?:mp4|mov|avi|m4v)$",
    re.IGNORECASE,
)

SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full VPR pipeline. Auto-skips stages that are already done."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every stage, ignoring auto-skip detection.",
    )
    for stage in ("sync", "extract", "annotate", "splits", "pipeline", "eval"):
        parser.add_argument(
            f"--force-{stage}",
            action="store_true",
            help=f"Force the '{stage}' stage even if its output is up to date.",
        )
        parser.add_argument(
            f"--skip-{stage}",
            action="store_true",
            help=f"Skip the '{stage}' stage entirely.",
        )
    parser.add_argument(
        "--min-videos",
        type=int,
        default=10,
        help=(
            "Threshold for the SYNC skip-detection: if data/raw_videos/ already "
            "has at least this many canonical .mov files, sync is skipped."
        ),
    )
    return parser.parse_args()


def run_step(label: str, command: list[str]) -> None:
    print()
    print("=" * 70)
    print(f"[run_all] STAGE: {label}")
    print(f"[run_all] cmd: {' '.join(command)}")
    print("=" * 70)
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(
            f"[run_all] STAGE '{label}' failed with exit code {result.returncode}."
        )


def list_canonical_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.glob("*")
        if path.is_file()
        and CANONICAL_VIDEO_REGEX.match(path.name)
        and path.stat().st_size > 0
    ]


def needs_sync(min_videos: int) -> tuple[bool, str]:
    canonical = list_canonical_videos(RAW_VIDEOS_DIR)
    if len(canonical) >= min_videos:
        return False, (
            f"{len(canonical)} canonical videos already on disk "
            f"(threshold {min_videos}). Pass --force-sync to pull anyway."
        )
    return True, f"only {len(canonical)} canonical videos on disk; running gdown sync."


def needs_extract() -> tuple[bool, str]:
    """Skip extract iff every canonical video already has frames on disk.

    We deliberately do NOT require the CSV to be 100% on disk — Ariel's
    floor10..16 rows reference frames he never shared, and that is an
    expected partial-data state. The signal that matters is "did ffmpeg
    already process every video that is currently in raw_videos/".
    """
    videos = list_canonical_videos(RAW_VIDEOS_DIR)
    if not videos:
        return True, "no canonical videos found in raw_videos/."

    if not DEFAULT_DATASET_CSV.exists():
        return True, "dataset.csv does not exist yet."

    videos_with_frames = 0
    missing = []
    for video in videos:
        # Determine the destination directory the extract script will use.
        # Above-ground (f<NN>) -> floor<N>/, basement (b<N>) -> basement<N>/.
        stem_lower = video.stem.lower()
        f_match = re.match(r"^f(\d{2})", stem_lower)
        b_match = re.match(r"^b(\d)", stem_lower)
        if f_match:
            floor_dir = PROCESSED_FRAMES_DIR / f"floor{int(f_match.group(1))}"
        elif b_match:
            floor_dir = PROCESSED_FRAMES_DIR / f"basement{b_match.group(1)}"
        else:
            # Not one of our canonical prefixes — extract will fail loudly,
            # which is what we want.
            continue

        # The extract script normalises the stem (lowercase, non-alphanumeric -> _);
        # compute the same normalisation so the prefix check is accurate.
        normalized_stem = re.sub(r"[^0-9a-zA-Z]+", "_", video.stem).strip("_").lower()
        if floor_dir.exists() and any(
            p.name.startswith(normalized_stem) for p in floor_dir.glob("*.jpg")
        ):
            videos_with_frames += 1
        else:
            missing.append(video.name)

    if not missing:
        return False, (
            f"{videos_with_frames}/{len(videos)} videos already have frames "
            "on disk."
        )
    return True, (
        f"{videos_with_frames}/{len(videos)} videos have frames; missing: "
        f"{', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}"
    )


def needs_annotate() -> tuple[bool, str]:
    """Skip if every CSV row already has area / section / floor_range filled."""
    if not DEFAULT_DATASET_CSV.exists():
        return True, "dataset.csv does not exist yet."
    df = pd.read_csv(DEFAULT_DATASET_CSV)
    needed_columns = ("area", "section", "floor_range")
    missing = [c for c in needed_columns if c not in df.columns]
    if missing:
        return True, f"columns missing: {', '.join(missing)}."
    empty = 0
    for column in needed_columns:
        empty += df[column].fillna("").astype(str).eq("").sum()
    if empty == 0:
        return False, (
            f"every one of {len(df)} rows already has area / section / floor_range."
        )
    return True, f"{empty} cells still empty across area / section / floor_range."


def needs_splits() -> tuple[bool, str]:
    if not DEFAULT_DATASET_CSV.exists():
        return True, "dataset.csv does not exist yet."
    df = pd.read_csv(DEFAULT_DATASET_CSV)
    if "split" not in df.columns:
        return True, "split column is missing."
    empty = df["split"].fillna("").astype(str).eq("").sum()
    if empty == 0:
        return False, f"every one of {len(df)} rows already has a split assignment."
    return True, f"{empty}/{len(df)} rows still have an empty split."


def needs_pipeline() -> tuple[bool, str]:
    embeddings = EMBEDDINGS_DIR / "gallery_embeddings.npy"
    metadata = EMBEDDINGS_DIR / "gallery_metadata.csv"
    index = INDEX_DIR / "gallery.index"
    missing = [str(p) for p in (embeddings, metadata, index) if not p.exists()]
    if not missing:
        return False, "embeddings + metadata + FAISS index already on disk."
    return True, f"missing artifact(s): {', '.join(missing)}"


def needs_eval() -> tuple[bool, str]:
    eval_path = RESULTS_DIR / "evaluation.json"
    if eval_path.exists():
        return False, f"{eval_path} already exists."
    return True, f"{eval_path} not produced yet."


def maybe_run(
    stage: str,
    label: str,
    command: list[str],
    *,
    needs_fn,
    force: bool,
    skip: bool,
    force_stage: bool,
) -> None:
    if skip:
        print(f"[run_all] STAGE: {label} -- SKIPPED (--skip-{stage}).")
        return
    if force or force_stage:
        print(f"[run_all] STAGE: {label} -- forced.")
        run_step(label, command)
        return
    needed, reason = needs_fn()
    if not needed:
        print(f"[run_all] STAGE: {label} -- SKIPPED ({reason}).")
        return
    print(f"[run_all] STAGE: {label} -- {reason}")
    run_step(label, command)


def main() -> None:
    args = parse_args()
    python = sys.executable

    maybe_run(
        "sync",
        "1/6 SYNC raw videos from Drive",
        [python, str(SCRIPTS_DIR / "sync_drive_data.py"), "--no-strict"],
        needs_fn=lambda: needs_sync(args.min_videos),
        force=args.force,
        skip=args.skip_sync,
        force_stage=args.force_sync,
    )
    maybe_run(
        "extract",
        "2/6 EXTRACT frames + update CSV",
        [python, str(SCRIPTS_DIR / "extract_frames_and_update_csv.py")],
        needs_fn=needs_extract,
        force=args.force,
        skip=args.skip_extract,
        force_stage=args.force_extract,
    )
    maybe_run(
        "annotate",
        "3/6 ANNOTATE area / section / floor_range",
        [python, str(SCRIPTS_DIR / "annotate_hierarchy.py")],
        needs_fn=needs_annotate,
        force=args.force,
        skip=args.skip_annotate,
        force_stage=args.force_annotate,
    )
    maybe_run(
        "splits",
        "4/6 ASSIGN gallery / query splits",
        [python, str(SCRIPTS_DIR / "assign_splits.py")],
        needs_fn=needs_splits,
        force=args.force,
        skip=args.skip_splits,
        force_stage=args.force_splits,
    )
    maybe_run(
        "pipeline",
        "5/6 EXTRACT embeddings + BUILD FAISS index",
        [python, str(SCRIPTS_DIR / "run_pipeline.py")],
        needs_fn=needs_pipeline,
        force=args.force,
        skip=args.skip_pipeline,
        force_stage=args.force_pipeline,
    )
    maybe_run(
        "eval",
        "6/6 EVALUATE retrieval quality",
        [python, str(SCRIPTS_DIR / "run_evaluation.py")],
        needs_fn=needs_eval,
        force=args.force,
        skip=args.skip_eval,
        force_stage=args.force_eval,
    )

    print()
    print("[run_all] All stages complete.")
    print("[run_all] Next steps:")
    print("  - Open the demo: marimo edit app/demo.py")
    print("  - Or the Jupyter notebook: jupyter notebook notebooks/test_model.ipynb")


if __name__ == "__main__":
    main()
