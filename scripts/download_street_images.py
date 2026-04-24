#!/usr/bin/env python3
"""Download center/left/right POV JPGs referenced in a VAIL annotations XML.

See download_street_images_design.md for rationale on layout, skip logic, and
retry policy. Paths resolve relative to this script's location so it works from
any CWD.
"""
from __future__ import annotations

import argparse
import os
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


# Paths resolve from the script's own location so the script works from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT.parent / "data"
DEFAULT_XML = (
    DATA_DIR
    / "route_annotations"
    / "Batch-5-US_5_St_Johnsbury_(Sameeh)(Ready_for_VTrans_Review)_merged.xml"
)
DEFAULT_OUT = DATA_DIR / "street_images"

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
    reference the same underlying JPG (bus slowed and the same frame was
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download center/left/right POV JPGs from a VAIL annotations XML."
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
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

    xml_path: Path = args.xml.expanduser().resolve()
    out_dir: Path = args.out.expanduser().resolve()

    if not xml_path.is_file():
        print(f"error: xml not found at {xml_path}", file=sys.stderr)
        return 2

    # Parse the XML into a flat list of FrameViews. --limit trims for smoke tests.
    all_frames = parse_frame_views(xml_path, args.views)
    if args.limit is not None:
        all_frames = all_frames[: args.limit]

    # Split into already-on-disk vs still-to-fetch. download_one also re-checks
    # per-worker; this pre-split exists so the plan summary is accurate.
    cached_frames: list[FrameView] = []
    frames_to_fetch: list[FrameView] = []
    for frame in all_frames:
        (cached_frames if is_cached(frame, out_dir) else frames_to_fetch).append(frame)

    # Print a one-line plan before any network traffic.
    count_by_view = {view: 0 for view in args.views}
    for frame in all_frames:
        count_by_view[frame.view] = count_by_view.get(frame.view, 0) + 1
    print(
        f"plan: {len(all_frames)} frames "
        f"({', '.join(f'{v}={n}' for v, n in count_by_view.items())}) | "
        f"{len(cached_frames)} cached | {len(frames_to_fetch)} to fetch"
    )

    if args.dry_run:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_partial_files(out_dir)

    # Counters shared across worker threads. counter_lock guards both.
    status_counts = {
        "downloaded": 0,
        "cached": len(cached_frames),
        "missing": 0,
        "failed": 0,
    }
    total_bytes = sum((out_dir / f.filename).stat().st_size for f in cached_frames)
    counter_lock = Lock()
    session = build_session()

    # Run many HTTP GETs concurrently. I/O-bound work → threads over processes.
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_frame = {
            executor.submit(download_one, session, frame, out_dir): frame
            for frame in frames_to_fetch
        }
        with tqdm(
            total=len(frames_to_fetch), unit="img", disable=not frames_to_fetch
        ) as progress_bar:
            for future in as_completed(future_to_frame):
                frame = future_to_frame[future]
                try:
                    status, bytes_written = future.result()
                except Exception as exc:
                    status, bytes_written = "failed", 0
                    tqdm.write(f"failed {frame.filename}: {exc}")

                with counter_lock:
                    status_counts[status] = status_counts.get(status, 0) + 1
                    total_bytes += bytes_written

                # Surface only problems to the terminal; successes show in the bar.
                if status in ("missing", "failed"):
                    tqdm.write(f"{status} {frame.filename} ({frame.url})")
                progress_bar.update(1)

    print(
        f"done: downloaded={status_counts['downloaded']} cached={status_counts['cached']} "
        f"missing={status_counts['missing']} failed={status_counts['failed']} "
        f"total_bytes={total_bytes}"
    )
    return 0 if status_counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
