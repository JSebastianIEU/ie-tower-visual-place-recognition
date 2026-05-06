"""Extract 1 FPS frames from raw videos and update ``data/metadata/dataset.csv``.

Two modes:

* ``--mode append`` (default): keeps every row already in the CSV, processes
  videos that match ``--floors-glob`` (default = all), and merges the new rows
  via ``concat + drop_duplicates`` so re-running is idempotent.
* ``--mode replace``: rebuilds the CSV from scratch using only the videos found
  in ``--videos-dir``. Useful for a clean local rerun, but **destroys** rows
  contributed by other team members.

Frame extraction itself is also idempotent: if the expected ``{stem}_frame_*.jpg``
files already exist for a video and ``--overwrite`` is not set, the video is
skipped.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extract_frames import extract_frames
from src.utils.config import (
    DATA_DIR,
    DEFAULT_DATASET_CSV,
    PROCESSED_FRAMES_DIR,
    RAW_VIDEOS_DIR,
)


SUPPORTED_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".m4v"}
CSV_COLUMNS = ["image_path", "label", "split", "device", "lighting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 1 FPS frames from videos and update dataset.csv."
    )
    parser.add_argument(
        "--videos-dir",
        default=str(RAW_VIDEOS_DIR),
        help=(
            "Directory that contains raw videos. The script walks it "
            "recursively, so subfolders like floors10-16/ or floors3-9/ work."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_FRAMES_DIR),
        help="Root directory where extracted frames will be saved.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_DATASET_CSV),
        help="Path to the dataset CSV that will be read and rewritten.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Number of frames to extract per second.",
    )
    parser.add_argument(
        "--mode",
        choices=("append", "replace"),
        default="append",
        help=(
            "append: keep existing CSV rows, merge new ones (default and "
            "recommended for shared work). replace: wipe the CSV and rebuild "
            "from videos in --videos-dir only."
        ),
    )
    parser.add_argument(
        "--floors-glob",
        default="*",
        help=(
            "Glob applied to the video stem (e.g. 'f0[3-9]_*' to only process "
            "floors 3-9). Default '*' processes every video found."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract frames even if files already exist on disk.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return normalized or "video"


def infer_floor_label(video_stem: str) -> str:
    """Infer the dataset label from the canonical filename prefix.

    Above-ground videos start with ``fNN_`` and produce ``floor<N>`` (no
    leading zero, matching Ariel's existing 879 rows). Basement videos
    start with ``bN_`` and produce ``basement<N>`` so they cannot collide
    with above-ground floors of the same number (e.g. ``b3_*`` is NOT
    ``floor3``).
    """
    stem_lower = video_stem.lower()
    match = re.match(r"^(f\d{2})", stem_lower)
    if match:
        # int() drops the leading zero: f03 -> 3 -> 'floor3', f10 -> 'floor10'.
        return f"floor{int(match.group(1)[1:])}"

    basement_match = re.match(r"^b(\d)", stem_lower)
    if basement_match:
        return f"basement{basement_match.group(1)}"

    raise ValueError(
        f"Could not infer floor from video name '{video_stem}'. "
        "Expected names to start with fNN (above-ground, e.g. f03_hallway_left) "
        "or bN (basement, e.g. b3_chill_lounge)."
    )


def list_video_paths(videos_dir: Path, glob_pattern: str) -> list[Path]:
    # Skip 0-byte files. sync_drive_data.py leaves these as placeholders so
    # gdown does not re-download videos that have already been canonicalised
    # — they are not real videos and ffmpeg fails on them with
    # "moov atom not found".
    paths = [
        path
        for path in videos_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        and path.stat().st_size > 0
    ]
    if glob_pattern and glob_pattern != "*":
        paths = [path for path in paths if fnmatch.fnmatch(path.stem.lower(), glob_pattern.lower())]
    return sorted(paths)


def build_image_path_for_csv(frame_path: Path, data_dir: Path) -> str:
    try:
        relative = frame_path.relative_to(data_dir)
    except ValueError:
        relative = frame_path
    return str(relative).replace("\\", "/")


def load_existing_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(csv_path)
    # Ensure every expected column exists (older CSVs may be missing optional ones).
    for column in CSV_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[CSV_COLUMNS]


def main() -> None:
    args = parse_args()
    data_dir = DATA_DIR.resolve()
    videos_dir = Path(args.videos_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    csv_path = Path(args.csv_path).resolve()

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    video_paths = list_video_paths(videos_dir, args.floors_glob)
    if not video_paths:
        raise ValueError(
            f"No videos found in {videos_dir} (after applying glob "
            f"'{args.floors_glob}')."
        )

    new_rows: list[dict[str, str]] = []
    floor_counts: dict[str, int] = {}

    for video_path in video_paths:
        floor_label = infer_floor_label(video_path.stem)
        floor_output_dir = output_dir / floor_label

        safe_video_stem = normalize_name(video_path.stem)
        image_pattern = f"{safe_video_stem}_frame_%06d.jpg"

        # Idempotency: if frames already exist for this video and the user did
        # not pass --overwrite, skip the ffmpeg call entirely.
        existing_frames = sorted(floor_output_dir.glob(f"{safe_video_stem}_frame_*.jpg"))
        if existing_frames and not args.overwrite:
            print(
                f"[skip] {video_path.name} -> {floor_output_dir} "
                f"({len(existing_frames)} frames already present)"
            )
            frame_paths = existing_frames
        else:
            extract_frames(
                video_path=video_path,
                output_dir=floor_output_dir,
                fps=args.fps,
                image_pattern=image_pattern,
                overwrite=args.overwrite,
            )
            frame_paths = sorted(
                floor_output_dir.glob(f"{safe_video_stem}_frame_*.jpg")
            )
            if not frame_paths:
                raise RuntimeError(f"No frames were extracted for: {video_path}")
            print(
                f"[done] {video_path.name} -> {floor_output_dir} "
                f"({len(frame_paths)} frames)"
            )

        for frame_path in frame_paths:
            new_rows.append(
                {
                    "image_path": build_image_path_for_csv(
                        frame_path=frame_path,
                        data_dir=data_dir,
                    ),
                    "label": floor_label,
                    "split": "",
                    "device": "",
                    "lighting": "",
                }
            )
        floor_counts[floor_label] = floor_counts.get(floor_label, 0) + len(frame_paths)

    new_df = pd.DataFrame(new_rows, columns=CSV_COLUMNS)

    if args.mode == "append":
        existing_df = load_existing_csv(csv_path)
        existing_count = len(existing_df)
        # When the same row exists in both (idempotency) we keep the EXISTING
        # one so values manually edited (split, device, lighting) survive.
        merged = (
            pd.concat([existing_df, new_df], ignore_index=True)
            .drop_duplicates(subset=["image_path"], keep="first")
            .sort_values(["label", "image_path"])
            .reset_index(drop=True)
        )
        truly_new = len(merged) - existing_count
        print()
        print(
            f"Mode: append | existing={existing_count} | new={truly_new} | "
            f"total_after_dedup={len(merged)}"
        )
        final_df = merged
    else:
        # mode == "replace"
        final_df = (
            new_df.drop_duplicates(subset=["image_path"], keep="last")
            .sort_values(["label", "image_path"])
            .reset_index(drop=True)
        )
        print()
        print(f"Mode: replace | total={len(final_df)}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(csv_path, index=False)

    print(f"Processed videos: {len(video_paths)}")
    print(f"Floor folders touched: {', '.join(sorted(floor_counts))}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
