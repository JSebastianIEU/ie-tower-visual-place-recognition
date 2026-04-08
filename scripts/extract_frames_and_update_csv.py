import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.extract_frames import extract_frames
from src.utils.config import DATA_DIR, DEFAULT_DATASET_CSV, PROCESSED_FRAMES_DIR, RAW_VIDEOS_DIR


SUPPORTED_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".m4v"}
CSV_COLUMNS = ["image_path", "label", "split", "device", "lighting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 1 FPS frames from all floor videos and rebuild dataset.csv."
    )
    parser.add_argument(
        "--videos-dir",
        default=str(RAW_VIDEOS_DIR / "floors10-16"),
        help="Directory that contains raw videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_FRAMES_DIR),
        help="Root directory where extracted frames will be saved.",
    )
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_DATASET_CSV),
        help="Path to output dataset CSV.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Number of frames to extract per second.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite frame files if they already exist.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value).strip("_").lower()
    return normalized or "video"


def infer_floor_label(video_stem: str) -> str:
    match = re.match(r"^(f\d{2})", video_stem.lower())
    if not match:
        raise ValueError(
            f"Could not infer floor from video name '{video_stem}'. "
            "Expected names to start with fNN (example: f10_...)."
        )
    floor_code = match.group(1)
    return f"floor{floor_code[1:]}"


def list_video_paths(videos_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def build_image_path_for_csv(frame_path: Path, data_dir: Path) -> str:
    try:
        relative = frame_path.relative_to(data_dir)
    except ValueError:
        relative = frame_path
    return str(relative).replace("\\", "/")


def main() -> None:
    args = parse_args()
    data_dir = DATA_DIR.resolve()
    videos_dir = Path(args.videos_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    csv_path = Path(args.csv_path).resolve()

    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")

    video_paths = list_video_paths(videos_dir)
    if not video_paths:
        raise ValueError(f"No videos found in: {videos_dir}")

    rows: list[dict[str, str]] = []
    floor_counts: dict[str, int] = {}

    for video_path in video_paths:
        floor_label = infer_floor_label(video_path.stem)
        floor_output_dir = output_dir / floor_label

        safe_video_stem = normalize_name(video_path.stem)
        image_pattern = f"{safe_video_stem}_frame_%06d.jpg"

        extract_frames(
            video_path=video_path,
            output_dir=floor_output_dir,
            fps=args.fps,
            image_pattern=image_pattern,
            overwrite=args.overwrite,
        )

        frame_paths = sorted(floor_output_dir.glob(f"{safe_video_stem}_frame_*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frames were extracted for: {video_path}")

        for frame_path in frame_paths:
            rows.append(
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
        print(
            f"[done] {video_path.name} -> {floor_output_dir} "
            f"({len(frame_paths)} frames)"
        )

    dataframe = pd.DataFrame(rows, columns=CSV_COLUMNS)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(csv_path, index=False)

    unique_labels = sorted(floor_counts.keys())
    print(f"\nProcessed videos: {len(video_paths)}")
    print(f"CSV rows written: {len(dataframe)}")
    print(f"Floor folders used: {', '.join(unique_labels)}")
    print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
