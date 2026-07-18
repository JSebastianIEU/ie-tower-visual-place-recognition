"""Download the processed frames from the GitHub release into ``data/processed_frames/``.

The frames are no longer committed to the repository — they ship as a release
asset instead (see README > Data & privacy). This script pulls that asset and
unpacks it, so a fresh clone is one command away from running the pipeline.

Behaviour:
- Skips the download entirely if the expected number of frames is already on
  disk, so re-running it (or ``run_all.py``) is idempotent.
- Streams the tarball to a temporary file, optionally checks its SHA-256, and
  extracts it under ``data/processed_frames/``.
- Uses only the standard library, so it adds no dependency to requirements.txt.

Usage::

    python scripts/fetch_data.py                 # default release tag
    python scripts/fetch_data.py --force         # re-download and overwrite
    python scripts/fetch_data.py --tag v1.0-dataset
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_FRAMES_DIR

REPO = "JSebastianIEU/ie-tower-visual-place-recognition"
DEFAULT_TAG = "v1.0-dataset"
ASSET_NAME = "ie-tower-frames-blurred.tar.gz"

# Number of JPEGs in the published dataset. Used only to decide whether the
# frames are already in place; --force overrides it.
EXPECTED_FRAMES = 2877


def count_frames(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.rglob("*.jpg"))


def asset_url(tag: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}/{ASSET_NAME}"


def download(url: str, dest: Path) -> None:
    print(f"[fetch] Downloading {url}")
    try:
        with urllib.request.urlopen(url) as response:  # noqa: S310 - fixed https URL
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            with dest.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(
                            f"\r[fetch] {done/1e6:7.1f} / {total/1e6:.1f} MB ({pct:5.1f} %)",
                            end="",
                            flush=True,
                        )
            print()
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"[error] Download failed ({exc.code} {exc.reason}).\n"
            f"[hint]  Confirm the release tag exists: "
            f"https://github.com/{REPO}/releases"
        ) from exc


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(tar_path: Path, dest: Path) -> None:
    """Extract ``tar_path`` under ``dest``, refusing entries that escape it."""
    dest.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            target = (resolved_dest / member.name).resolve()
            if not target.is_relative_to(resolved_dest):
                raise SystemExit(f"[error] Refusing unsafe tar entry: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise SystemExit(f"[error] Refusing non-regular tar entry: {member.name}")
        tar.extractall(dest)  # noqa: S202 - members validated above


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the processed frames release asset into data/processed_frames/.",
    )
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Release tag to pull from.")
    parser.add_argument(
        "--output-dir",
        default=str(PROCESSED_FRAMES_DIR),
        help="Directory to extract the frames into.",
    )
    parser.add_argument(
        "--sha256",
        default=None,
        help="Expected SHA-256 of the tarball. Verified before extraction when given.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite even if frames are already present.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    present = count_frames(output_dir)
    if present >= EXPECTED_FRAMES and not args.force:
        print(f"[fetch] {present} frames already in {output_dir} — nothing to do.")
        print("[fetch] Re-run with --force to download again.")
        return 0
    if present:
        print(f"[fetch] Found {present} frames (expected {EXPECTED_FRAMES}) — fetching.")

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / ASSET_NAME
        download(asset_url(args.tag), tarball)

        if args.sha256:
            actual = sha256(tarball)
            if actual.lower() != args.sha256.lower():
                print(f"[error] SHA-256 mismatch.\n  expected {args.sha256}\n  actual   {actual}", file=sys.stderr)
                return 1
            print("[fetch] SHA-256 OK.")

        if args.force and output_dir.exists():
            shutil.rmtree(output_dir)

        print(f"[fetch] Extracting into {output_dir}")
        safe_extract(tarball, output_dir)

    final = count_frames(output_dir)
    print()
    print("=== Fetch summary ===")
    print(f"Frames on disk : {final}")
    print(f"Expected       : {EXPECTED_FRAMES}")
    print(f"Location       : {output_dir}")

    if final != EXPECTED_FRAMES:
        print(
            f"[warning] Expected {EXPECTED_FRAMES} frames but found {final}.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
