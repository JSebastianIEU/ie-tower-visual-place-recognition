from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 1,
    image_pattern: str = "frame_%06d.jpg",
    overwrite: bool = True,
) -> Path:
    """Extract frames from a video with ffmpeg."""

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        homebrew_ffmpeg = Path("/opt/homebrew/bin/ffmpeg")
        if homebrew_ffmpeg.exists():
            ffmpeg_path = str(homebrew_ffmpeg)

    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required but was not found in PATH. Install ffmpeg first."
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / image_pattern
    overwrite_flag = "-y" if overwrite else "-n"
    command = [
        ffmpeg_path,
        overwrite_flag,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(output_pattern),
    ]

    subprocess.run(command, check=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a single video.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--output", required=True, help="Directory to save frames.")
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="Number of frames to extract per second.",
    )
    parser.add_argument(
        "--image-pattern",
        default="frame_%06d.jpg",
        help="ffmpeg output image pattern.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = extract_frames(
        video_path=args.video,
        output_dir=args.output,
        fps=args.fps,
        image_pattern=args.image_pattern,
        overwrite=not args.no_overwrite,
    )
    print(f"Saved frames to: {output_dir}")


if __name__ == "__main__":
    main()
