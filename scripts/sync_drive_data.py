"""Sync raw videos from the team Google Drive into ``data/raw_videos/``.

Behaviour:
- Calls ``gdown.download_folder`` against the shared Drive folder. ``gdown``
  internally skips files whose name already exists locally, so re-running the
  script is incremental and idempotent.
- After downloading we walk ``output_dir`` and (by default) **rename every
  video to the canonical** ``fNN_<area>_<descriptor>.<ext>`` **convention**.
  The mapping is intentionally tolerant of the legacy naming Farah used in
  the team Drive (e.g. ``elevators_floor7.mov``, ``hallway_floor3 copy.mov``,
  ``classroom_4.01_floor4.mov``).
- After renaming we **leave a 0-byte placeholder at the legacy filename** so
  that the next ``gdown`` sync sees the legacy name on disk and skips the
  re-download. This avoids burning bandwidth re-pulling already-renamed
  videos.
- Anything we cannot map is reported and (by default) the script exits with
  code 2 so users notice immediately.
- ``--dry-run`` skips the download and only inspects/reports current state.
- ``--no-canonicalize`` disables the rename step (useful for debugging).

The Drive folder must be shared as "anyone with the link can view" for
``gdown`` to be able to enumerate it without OAuth.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_VIDEOS_DIR


DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/1b37B-V67FRRttLNrbHQ0bk4uywsZpiDH"
)

SUPPORTED_VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".m4v"}

# Canonical filename:
#   above-ground floors -> ``f<NN>_<area>_<descriptor...>.<ext>`` (NN = 2 digits)
#   basement floors     -> ``b<N>_<area>_<descriptor...>.<ext>``  (N  = 1 digit)
NAMING_REGEX = re.compile(
    r"^(?:f\d{2}|b\d)_[a-z0-9]+(?:_[a-z0-9]+)+\.(?:mp4|mov|avi|m4v)$",
    re.IGNORECASE,
)

# Map legacy area names (left) to the canonical token Ariel already used (right).
AREA_ALIASES = {
    "elevators": "elevator",
    "elevator": "elevator",
    "center": "central",
    "central": "central",
    "class": "classroom",
    "classroom": "classroom",
    "classrom": "classroom",  # typo in Ariel's CSV, kept as alias on read
    "hallway": "hallway",
    "stairs": "stairs",
    "open_area": "open_area",
    "openarea": "open_area",
    # cafeteria == central (the f22 cafeteria is the same open central space).
    # Keep this alias even though the local file was renamed manually, so that
    # future Drive uploads named ``cafeteria_*`` collapse correctly.
    "cafeteria": "central",
    # Study rooms / meeting rooms (small enclosed work areas, distinct from
    # classrooms). f17 and f22 originally tagged 'classroom' actually belong
    # to this group — confirm in Drive and rename if needed.
    "studyroom": "studyroom",
    "study_room": "studyroom",
    "studyrooms": "studyroom",
    "meetingroom": "meetingroom",
    "meeting_room": "meetingroom",
    "meetingrooms": "meetingroom",
}


def canonicalize_name(legacy_name: str) -> Optional[str]:
    """Map a legacy Drive video filename to canonical ``fNN_<area>_<descriptor>.<ext>``.

    Handles every pattern observed in the team Drive as of 2026-05-06::

        hallway_floor3.mov           -> f03_hallway_main.mov
        hallway_floor3 copy.mov      -> f03_hallway_alt.mov
        hallway_floor9 copy 2.mov    -> f09_hallway_alt2.mov
        elevators_floor3.mov         -> f03_elevator_main.mov
        elevators_flooor21.mov       -> f21_elevator_main.mov   (typo)
        elevators\\-floor7.mov        -> f07_elevator_main.mov   (backslash)
        center_floor4.mov            -> f04_central_main.mov
        class_floor22.mov            -> f22_classroom_main.mov
        cafeteria_floor22.mov        -> f22_cafeteria_main.mov
        classroom_4.01_floor4.mov    -> f04_classroom_4_01.mov

    Also handles Ayo's basement uploads with a camera/clip suffix::

        b3_chill_lounge__A001_04081241_C029.mov  -> b3_chill_lounge_c029.mov
        b2_hallway_walkpath__A001_04081244_C025.MOV -> b2_hallway_walkpath_c025.mov
        b0_basement_lift__A001_04081250_C030.MOV -> b0_basement_lift_c030.mov

    Returns ``None`` if the name does not look like a video or cannot be mapped.
    """
    stem, dot, ext_raw = legacy_name.rpartition(".")
    if not dot:
        return None
    ext = "." + ext_raw.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        return None

    # Already canonical (with possibly uppercase extension)? Just lowercase
    # the extension and return — the function is idempotent.
    if NAMING_REGEX.match(legacy_name):
        return stem.lower() + ext

    # Ayo's basement uploads: ``b<N>_<area>_<descriptor>__A001_<timestamp>_C<NNN>``.
    # The clip ID (`C029`, `C035`, ...) is the only thing that makes two
    # captures of the same area distinguishable, so we preserve it as the
    # last descriptor token. The rest of the camera metadata is dropped.
    basement_match = re.match(
        r"^(b\d)_(.+?)__A001_\d+_C(\d+)\s*$",
        stem,
        re.IGNORECASE,
    )
    if basement_match:
        floor_token = basement_match.group(1).lower()
        area_descriptor = basement_match.group(2).lower().strip("_- ")
        clip_id = basement_match.group(3).lower()
        return f"{floor_token}_{area_descriptor}_c{clip_id}{ext}"

    # Detect "copy" / "copy 2" suffix on the stem.
    copy_match = re.search(r"\s+copy(?:\s+(\d+))?\s*$", stem, re.IGNORECASE)
    if copy_match:
        copy_num = copy_match.group(1)
        descriptor = f"alt{copy_num}" if copy_num else "alt"
        stem = stem[: copy_match.start()].rstrip()
    else:
        descriptor = "main"

    # Pull the floor number. Tolerate typos like "flooor21" by accepting two or
    # more "o" characters between f/l and r.
    floor_match = re.search(r"f?l?o{2,}r(\d{1,2})", stem, re.IGNORECASE) or \
        re.search(r"floor(\d{1,2})", stem, re.IGNORECASE)
    if not floor_match:
        return None
    floor_num = int(floor_match.group(1))

    # Whatever is before the "floor..." token is the area part.
    area_part = stem[: floor_match.start()].rstrip("_- \\")

    # Special: "classroom_4.01" -> area=classroom, descriptor=4_01
    sub_match = re.match(r"^([a-zA-Z]+)_(\d+(?:[._]\d+)+)$", area_part)
    if sub_match:
        area = sub_match.group(1)
        descriptor = sub_match.group(2).replace(".", "_")
    else:
        area = area_part

    area_key = re.sub(r"[\\\-\s]+", "_", area).strip("_").lower()
    canonical_area = AREA_ALIASES.get(area_key, area_key)

    if not canonical_area:
        return None

    return f"f{floor_num:02d}_{canonical_area}_{descriptor}{ext}"


def _promote_orphan_alts(root: Path, summary: dict, dry_run: bool) -> None:
    """If a (floor, area) has an ``_alt`` but no ``_main``, promote the first
    ``_alt`` to ``_main``. Happens for floors where the only Drive video had
    " copy" baked into its name (e.g. floor 20's hallway video).
    """
    grouped: dict[tuple[str, str], list[Path]] = {}
    pattern = re.compile(r"^(f\d{2})_([a-z0-9]+)_([a-z0-9_]+)\.[a-z0-9]+$")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        if path.stat().st_size == 0:
            continue
        match = pattern.match(path.name.lower())
        if not match:
            continue
        floor_token, area, _descriptor = match.groups()
        grouped.setdefault((floor_token, area), []).append(path)

    for (floor_token, area), paths in grouped.items():
        names = [p.name.lower() for p in paths]
        has_main = any(f"_{area}_main." in name for name in names)
        if has_main:
            continue
        # Pick the alt with the lowest suffix number (alt < alt2 < alt3 ...).
        def alt_index(p: Path) -> int:
            m = re.search(r"_alt(\d*)\.", p.name)
            if not m:
                return 99
            digits = m.group(1)
            return int(digits) if digits else 1
        target = sorted(paths, key=alt_index)[0]
        new_name = re.sub(
            r"_alt\d*(\.[a-z0-9]+)$",
            r"_main\1",
            target.name,
            flags=re.IGNORECASE,
        )
        new_path = target.with_name(new_name)
        if new_path.exists():
            continue  # very unlikely race; skip safely.
        if dry_run:
            summary["renamed"].append(
                f"would-promote {target.name} -> {new_name} (no _main peer)"
            )
            continue
        target.rename(new_path)
        summary["renamed"].append(f"promote {target.name} -> {new_name}")


def canonicalize_directory(root: Path, dry_run: bool = False) -> dict:
    """Rename every legacy-named video under ``root`` to the canonical convention.

    Skips files that already match ``NAMING_REGEX`` and 0-byte placeholders left
    by previous syncs. After a rename we drop a 0-byte placeholder at the
    original path so future ``gdown`` syncs do not re-download the same video.

    Returns a dict with keys ``renamed``, ``deleted_duplicates``, ``unmappable``
    (each a list of human-readable strings) plus ``placeholders`` (count).
    """
    summary = {
        "renamed": [],
        "deleted_duplicates": [],
        "unmappable": [],
        "placeholders": 0,
    }

    if not root.exists():
        return summary

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
            continue
        if path.stat().st_size == 0:
            # Placeholder kept around so gdown's name-based skip works.
            summary["placeholders"] += 1
            continue
        if NAMING_REGEX.match(path.name):
            continue

        canonical_name = canonicalize_name(path.name)
        if canonical_name is None:
            summary["unmappable"].append(str(path))
            continue

        canonical_path = path.with_name(canonical_name)
        if canonical_path.exists() and canonical_path.stat().st_size > 0:
            # We already have the canonical version from a previous sync. The
            # legacy file is a wasted re-download; drop it.
            if dry_run:
                summary["deleted_duplicates"].append(
                    f"would-delete {path.name} (canonical already present)"
                )
            else:
                path.unlink()
                summary["deleted_duplicates"].append(
                    f"deleted {path.name} (canonical already present)"
                )
            continue

        if dry_run:
            summary["renamed"].append(f"would-rename {path.name} -> {canonical_name}")
            continue

        path.rename(canonical_path)
        # Leave a 0-byte placeholder so gdown skips this name on next sync.
        try:
            path.touch(exist_ok=False)
        except FileExistsError:  # pragma: no cover - defensive
            pass
        summary["renamed"].append(f"{path.name} -> {canonical_name}")

    _promote_orphan_alts(root, summary, dry_run=dry_run)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync raw videos from the team Drive into data/raw_videos/.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RAW_VIDEOS_DIR),
        help="Directory where raw videos will be saved.",
    )
    parser.add_argument(
        "--drive-url",
        default=DRIVE_FOLDER_URL,
        help="Override the shared Drive folder URL.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the download and only validate / report current state.",
    )
    parser.add_argument(
        "--no-canonicalize",
        action="store_true",
        help="Do NOT rename legacy filenames after download. The script will then "
        "treat any non-canonical filename as a hard error.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not exit with a non-zero status when files are mis-named.",
    )
    return parser.parse_args()


def list_videos(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )


def validate_names(videos: list[Path]) -> tuple[list[Path], list[Path]]:
    matched: list[Path] = []
    bad: list[Path] = []
    for video in videos:
        if video.stat().st_size == 0:
            # 0-byte placeholders left after canonical rename — ignore.
            continue
        if NAMING_REGEX.match(video.name):
            matched.append(video)
        else:
            bad.append(video)
    return matched, bad


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    before = {video.name for video in list_videos(output_dir)}

    if args.dry_run:
        print(f"[dry-run] Skipping download. Inspecting {output_dir}.")
    else:
        try:
            import gdown  # type: ignore[import-untyped]
        except ImportError:  # pragma: no cover - import guard
            print(
                "[error] gdown is not installed. Run `pip install -r requirements.txt`.",
                file=sys.stderr,
            )
            return 1

        print(f"[sync] Downloading from {args.drive_url}")
        print(f"[sync] Target directory: {output_dir}")
        try:
            # gdown 6.x removed `remaining_ok`; folders with 50+ files now
            # require iterating over a returned listing instead. We pass the
            # arguments compatible with both 5.x and 6.x: url, output,
            # quiet, use_cookies, resume.
            gdown.download_folder(
                url=args.drive_url,
                output=str(output_dir),
                quiet=False,
                use_cookies=False,
                resume=True,
            )
        except Exception as exc:  # noqa: BLE001 - gdown raises plain Exception
            print(f"[error] gdown failed: {exc}", file=sys.stderr)
            print(
                "[hint] Confirm the folder is shared as 'anyone with the link can "
                "view'. If Drive is rate-limiting, wait a few minutes and retry.",
                file=sys.stderr,
            )
            return 1

    if not args.no_canonicalize:
        print()
        print("[canonicalize] Renaming legacy filenames to fNN_<area>_<descriptor>.<ext>")
        rename_summary = canonicalize_directory(output_dir, dry_run=args.dry_run)
        for line in rename_summary["renamed"]:
            print(f"  rename: {line}")
        for line in rename_summary["deleted_duplicates"]:
            print(f"  cleanup: {line}")
        if rename_summary["unmappable"]:
            print()
            print("[warning] Could not map these files automatically:")
            for line in rename_summary["unmappable"]:
                print(f"  {line}")

    after_videos = list_videos(output_dir)
    after_names = {video.name for video in after_videos}

    new_count = len(after_names - before)
    existing_count = len(after_names & before)

    matched, bad = validate_names(after_videos)

    by_floor: dict[str, int] = {}
    for video in matched:
        floor = video.name[:3].lower()  # f03 / f10 / ...
        by_floor[floor] = by_floor.get(floor, 0) + 1

    print()
    print("=== Sync summary ===")
    print(f"Downloaded new : {new_count}")
    print(f"Already present: {existing_count}")
    print(f"Well-named     : {len(matched)}")
    print(f"Bad-named      : {len(bad)}")

    if by_floor:
        print()
        print("Videos by floor prefix:")
        for floor in sorted(by_floor):
            print(f"  {floor}: {by_floor[floor]}")

    if bad:
        print()
        print("Files that do not match the canonical naming convention:")
        for video in bad:
            print(f"  {video}")
        print()
        print(
            "Fix these by renaming them in Google Drive to "
            "fNN_<area>_<descriptor>.<ext> (lowercase). "
            "See README.md > Team Data Ingestion."
        )
        if not args.no_strict:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
