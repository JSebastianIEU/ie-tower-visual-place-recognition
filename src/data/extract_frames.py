from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _windows_short_path(path: Path) -> str:
    """Return the Windows 8.3 short-path for ``path`` if its long form contains
    non-ASCII characters; otherwise return the long form unchanged.

    Why we need this: ``ffmpeg`` on Windows uses ``fopen`` (narrow char) to
    open input and output files. When the path goes through a username with
    accents (e.g. ``C:\\Users\\Juan Sebastian Peña\\…``), the encoding round
    trip mangles the bytes and ffmpeg fails with "No such file or directory".

    Windows automatically maintains an ASCII-safe 8.3 alias for every NTFS
    path (``C:\\Users\\JUANSE~1\\…``). The alias points at the same files —
    we just have to ask Windows for it via ``GetShortPathNameW``. Nothing
    on the user's filesystem changes.
    """
    full = str(path)
    if sys.platform != "win32" or full.isascii():
        return full
    if not Path(full).exists():
        # GetShortPathNameW only works for paths that exist. The caller
        # creates the output directory before invoking us, and the input
        # video is checked above, so we should not hit this branch in
        # practice — fall back to the long path if we somehow do.
        return full
    try:
        import ctypes
        from ctypes import wintypes

        get_short = ctypes.windll.kernel32.GetShortPathNameW
        get_short.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        get_short.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(1024)
        n = get_short(full, buf, 1024)
        if 0 < n < 1024:
            return buf.value
    except Exception:  # noqa: BLE001 - safety net; fall through to long path.
        pass
    return full


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
        # Fallback: ``imageio-ffmpeg`` (a runtime dependency) ships a static
        # ffmpeg binary so the pipeline runs even when no system ffmpeg is
        # installed. Used by team members on Windows / locked-down machines.
        try:
            import imageio_ffmpeg  # type: ignore[import-not-found]

            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required but was not found. Install it system-wide "
            "(brew/choco/apt) or `pip install imageio-ffmpeg` for a bundled "
            "binary."
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert non-ASCII paths to their Windows 8.3 short form so ffmpeg's
    # narrow-char fopen can open them. The image pattern is pure ASCII
    # (e.g. ``f03_hallway_main_frame_%06d.jpg``) so it can be appended to
    # the short directory verbatim. The ffmpeg binary path likewise lives
    # under the user profile and is short-pathed for the same reason.
    ffmpeg_arg = _windows_short_path(Path(ffmpeg_path))
    video_arg = _windows_short_path(video_path)
    output_pattern_arg = (
        Path(_windows_short_path(output_dir)) / image_pattern
    )

    overwrite_flag = "-y" if overwrite else "-n"
    command = [
        ffmpeg_arg,
        overwrite_flag,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_arg,
        "-vf",
        f"fps={fps}",
        str(output_pattern_arg),
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
