#!/usr/bin/env python3
"""Download center/left/right POV JPGs referenced in a VAIL annotations XML.

See download_street_images_design.md for rationale on layout, skip logic, and
retry policy. Paths resolve relative to this script's location so it works from
any CWD.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import requests
from tqdm import tqdm

# Put the repo root on the path so `config` is importable when run as a script.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths  # noqa: E402

# Inbox/archive workflow: drop new XMLs into new_routes/, the script processes
# them and moves only the fully-downloaded ones into onboarded_routes/.
DEFAULT_INBOX = paths.NEW_ROUTES_DIR
DEFAULT_ARCHIVE = paths.ONBOARDED_ROUTES_DIR
DEFAULT_OUT = paths.STREET_IMAGES_DIR

# HTTP tuning. BACKOFF_SECONDS is the sleep between retry attempts.
CHUNK_BYTES = 64 * 1024
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30
BACKOFF_SECONDS = (1, 2, 4)

# The three camera views each annotation exposes under <source>.
VIEWS = ("center", "left", "right")


@dataclass(frozen=True)
class FrameView:
    """One image to download — a single view of a single frame."""
    filename: str
    url: str
    view: str
    batch: str


def parse_frame_views(xml_path: Path, views: tuple[str, ...]) -> list[FrameView]:
    """Walk the XML and return one FrameView per (annotation, requested view).

    Deduplicates by filename across the whole batch — two annotations that
    reference the same underlying JPG (source vehicle slowed and the same frame was
    captured twice) collapse into a single download.
    """
    batch_name = xml_path.stem
    root = ET.parse(xml_path).getroot()
    frames_by_filename: dict[str, FrameView] = {}

    for annotation in root.findall("annotation"):
        source = annotation.find("source")
        if source is None:
            continue
        for view in views:
            url = (source.findtext(f"src_{view}") or "").strip()
            if not url or url.lower() == "none":
                continue
            # Use the URL basename as the local filename. Center/left/right each
            # embed a distinct token (_ROW_ / _ForwardLeft_ / _ForwardRight_),
            # so flat storage doesn't collide across views.
            filename = Path(urlsplit(url).path).name
            if not filename or filename in frames_by_filename:
                continue
            frames_by_filename[filename] = FrameView(
                filename=filename, url=url, view=view, batch=batch_name
            )
    return list(frames_by_filename.values())


def is_cached(frame: FrameView, out_dir: Path) -> bool:
    """True if a non-empty file already sits at the destination path."""
    dest = out_dir / frame.filename
    return dest.exists() and dest.stat().st_size > 0


def download_one(
    session: requests.Session, frame: FrameView, out_dir: Path
) -> tuple[str, int]:
    """Fetch one FrameView to disk. Returns (status, bytes_written).

    Streams to a `.part` file and renames on success, so a Ctrl-C can't leave
    a half-written file that looks cached next run. Retries on 5xx / connection
    errors; non-404 4xx is treated as a permanent failure and not retried.
    """
    dest = out_dir / frame.filename
    if is_cached(frame, out_dir):
        return "cached", dest.stat().st_size

    partial_path = dest.with_suffix(dest.suffix + ".part")
    if partial_path.exists():
        partial_path.unlink()

    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            with session.get(
                frame.url, stream=True, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            ) as response:
                if response.status_code == 404:
                    return "missing", 0
                if 400 <= response.status_code < 500:
                    return "failed", 0
                response.raise_for_status()

                # Stream to disk in chunks so 4K JPGs never live fully in memory.
                bytes_written = 0
                with open(partial_path, "wb") as out_file:
                    for chunk in response.iter_content(CHUNK_BYTES):
                        if chunk:
                            out_file.write(chunk)
                            bytes_written += len(chunk)

            os.replace(partial_path, dest)  # atomic on the same filesystem
            return "downloaded", bytes_written
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt < len(BACKOFF_SECONDS):
                time.sleep(BACKOFF_SECONDS[attempt])
            continue

    if partial_path.exists():
        partial_path.unlink()
    return "failed", 0


# Weird stuff used by multiprocesing in python
def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "poleulizer-downloader/0.2"})
    return session


def sweep_partial_files(out_dir: Path) -> None:
    """Remove any .part leftovers from a previously interrupted run."""
    if not out_dir.exists():
        return
    for partial in out_dir.glob("*.part"):
        partial.unlink()


def parse_views(raw: str) -> tuple[str, ...]:
    requested = tuple(v.strip().lower() for v in raw.split(",") if v.strip())
    unknown = [v for v in requested if v not in VIEWS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown view(s): {unknown}. allowed: {VIEWS}"
        )
    return requested or VIEWS


def process_xml(
    xml_path: Path,
    out_dir: Path,
    views: tuple[str, ...],
    limit: int | None,
    max_workers: int,
    session: requests.Session,
) -> dict[str, int]:
    """Download every frame referenced by one XML. Returns per-status counts.

    Counts include `downloaded`, `cached`, `missing`, `failed`, and `bytes`.
    The XML is considered fully processed only when `missing == failed == 0`;
    the caller uses that to decide whether to archive it.
    """
    all_frames = parse_frame_views(xml_path, views)
    if limit is not None:
        all_frames = all_frames[:limit]

    cached_frames: list[FrameView] = []
    frames_to_fetch: list[FrameView] = []
    for frame in all_frames:
        (cached_frames if is_cached(frame, out_dir) else frames_to_fetch).append(frame)

    count_by_view = {view: 0 for view in views}
    for frame in all_frames:
        count_by_view[frame.view] = count_by_view.get(frame.view, 0) + 1
    print(
        f"  plan: {len(all_frames)} frames "
        f"({', '.join(f'{v}={n}' for v, n in count_by_view.items())}) | "
        f"{len(cached_frames)} cached | {len(frames_to_fetch)} to fetch"
    )

    counts = {
        "downloaded": 0,
        "cached": len(cached_frames),
        "missing": 0,
        "failed": 0,
        "bytes": sum(
            (out_dir / f.filename).stat().st_size for f in cached_frames
        ),
    }

    if not frames_to_fetch:
        return counts

    counter_lock = Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {
            executor.submit(download_one, session, frame, out_dir): frame
            for frame in frames_to_fetch
        }
        # tqdm desc is truncated to keep multi-XML output readable.
        bar_desc = xml_path.name[:30]
        with tqdm(
            total=len(frames_to_fetch), unit="img", desc=bar_desc
        ) as progress_bar:
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                try:
                    status, bytes_written = future.result()
                except Exception as exc:
                    status, bytes_written = "failed", 0
                    tqdm.write(f"failed {frame.filename}: {exc}")

                with counter_lock:
                    counts[status] = counts.get(status, 0) + 1
                    counts["bytes"] += bytes_written

                # Surface only problems to the terminal; successes show in the bar.
                if status in ("missing", "failed"):
                    tqdm.write(f"{status} {frame.filename} ({frame.url})")
                progress_bar.update(1)

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download center/left/right POV JPGs from VAIL annotation XMLs."
    )
    # Two operating modes:
    #   default: process every *.xml in --inbox, archive successful ones
    #   --xml:   one-off mode for ad-hoc processing, archive disabled
    parser.add_argument(
        "--xml",
        type=Path,
        default=None,
        help="process a single XML (override; archive move is disabled in this mode)",
    )
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--views",
        type=parse_views,
        default=VIEWS,
        help=f"comma-separated subset of {VIEWS} (default: all)",
    )
    args = parser.parse_args()

    out_dir: Path = args.out.expanduser().resolve()

    # Resolve which XMLs to process and whether to archive them after.
    if args.xml is not None:
        xml_path = args.xml.expanduser().resolve()
        if not xml_path.is_file():
            print(f"error: xml not found at {xml_path}", file=sys.stderr)
            return 2
        xml_files = [xml_path]
        archive_dir: Path | None = None  # ad-hoc mode: never auto-move
    else:
        inbox_dir = args.inbox.expanduser().resolve()
        archive_dir = args.archive.expanduser().resolve()
        if not inbox_dir.is_dir():
            print(f"error: inbox dir not found at {inbox_dir}", file=sys.stderr)
            return 2
        xml_files = sorted(inbox_dir.glob("*.xml"))
        if not xml_files:
            print(f"nothing to process: no XML files in {inbox_dir}. Download .xml files using Maplyzer UI and upload to this directory.")
            return 0

    if args.dry_run:
        # Per-file plan, no HTTP. Useful for confirming the inbox is what you expect.
        for xml_path in xml_files:
            all_frames = parse_frame_views(xml_path, args.views)
            if args.limit is not None:
                all_frames = all_frames[: args.limit]
            cached = sum(1 for f in all_frames if is_cached(f, out_dir))
            print(
                f"  {xml_path.name}: {len(all_frames)} frames | "
                f"{cached} cached | {len(all_frames) - cached} to fetch"
            )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_partial_files(out_dir)
    if archive_dir is not None:
        archive_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    print(f"processing {len(xml_files)} XML file(s) from inbox")

    aggregate = {"downloaded": 0, "cached": 0, "missing": 0, "failed": 0, "bytes": 0}
    archived_count = 0
    kept_count = 0

    for xml_path in xml_files:
        print(f"\n[{xml_path.name}]")
        counts = process_xml(
            xml_path, out_dir, args.views, args.limit, args.max_workers, session
        )
        for k in aggregate:
            aggregate[k] += counts.get(k, 0)

        # Only archive when every referenced image is on disk. Otherwise log
        # an error and leave the XML in the inbox so the next run retries.
        if archive_dir is not None:
            all_present = counts["missing"] == 0 and counts["failed"] == 0
            if all_present:
                target = archive_dir / xml_path.name
                if target.exists():
                    print(
                        f"  ERROR: archive already contains {target.name}; "
                        f"leaving original in inbox to avoid silent overwrite",
                        file=sys.stderr,
                    )
                    kept_count += 1
                else:
                    shutil.move(str(xml_path), str(target))
                    print(f"  archived -> {archive_dir.name}/{xml_path.name}")
                    archived_count += 1
            else:
                print(
                    f"  ERROR: {counts['missing']} missing + {counts['failed']} failed "
                    f"download(s) for {xml_path.name}; leaving in inbox for retry",
                    file=sys.stderr,
                )
                kept_count += 1

    print(
        f"\ndone: downloaded={aggregate['downloaded']} cached={aggregate['cached']} "
        f"missing={aggregate['missing']} failed={aggregate['failed']} "
        f"total_bytes={aggregate['bytes']}"
    )
    if archive_dir is not None:
        print(f"      archived={archived_count} kept_in_inbox={kept_count}")

    # Non-zero if any XML stayed in the inbox (incomplete download or collision).
    if archive_dir is not None and kept_count > 0:
        return 1
    return 0 if aggregate["failed"] == 0 and aggregate["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
