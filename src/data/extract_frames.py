import subprocess
from pathlib import Path


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: int = 1,
    image_pattern: str = "frame_%06d.jpg",
) -> Path:
    """Extract frames from a video with ffmpeg."""

    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = output_dir / image_pattern
    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(output_pattern),
    ]

    subprocess.run(command, check=True)
    return output_dir
