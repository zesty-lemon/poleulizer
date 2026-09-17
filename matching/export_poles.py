"""Export a pole-centric deliverable from the matcher output.

Reorganizes pole_box_matches around the physical pole so you can go to a real
`pole_id` and see every image it appears in and which bounding box IS the pole in
each. Produces, under an export folder:

    poles_index.csv    one row per (pole, image): pole_id -> image -> box_index
    poles_summary.csv  one row per pole: coverage counts + frame span
    by_pole/<pole>/    that pole's matched frames, its box highlighted green

Read-only: consumes pole_box_matches.parquet + bboxes.parquet + the street images.
No model, no re-matching. See documentation/pole_export_design.md.

Run:
    python3 -m matching.export_poles                    # CSV + per-pole images
    python3 -m matching.export_poles --no-images        # CSV only (fast)
    python3 -m matching.export_poles --matched-only      # drop unmatched CSV rows
    python3 -m matching.export_poles --crops             # also save tight box crops
    python3 -m matching.export_poles --subset ../data/matching/isolated_poles.csv \\
        --out ../data/matching/pole_export_isolated     # only the isolated poles
                                                        # (subset CSV from find_isolated_poles)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from matching.find_isolated_poles import run_find_isolated_poles_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths  # noqa: E402
from detection.run_detection import load_detections  # noqa: E402
from matching.matching_pipeline import load_matches  # noqa: E402
from matching.visualize_matches import (  # noqa: E402
    _box_corners_for_frame, _load_font,
    MATCHED_COLOR, UNMATCHED_COLOR, ORPHAN_COLOR,
    MATCHED_BOX_WIDTH, ORPHAN_BOX_WIDTH, UNMATCHED_LINE_WIDTH,
    FONT_SIZE, LABEL_ABOVE_BOX_PX,
)

# Default output roots (kept in-script rather than config.paths to stay self-contained).
DEFAULT_EXPORT_DIR = paths.MATCHING_DIR / "pole_export"                   # whole catalog
ISOLATED_EXPORT_DIR = paths.MATCHING_DIR / "pole_export_isolated"         # isolated subset
CROP_MARGIN_PX = 60  # padding around a box when saving a --crops thumbnail

BBOX_COLS = ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]


# ----- naming (filesystem-safe views of the real pole_id) -----

def _geohash_suffix(pole_id: str) -> str:
    """`geohash:druu9hynx` -> `druu9hynx` (drop the colon; creates a path hazard on macOS)."""
    return str(pole_id).split(":")[-1]


def _pole_folder_name(pole_index, pole_id) -> str:
    """collision-free folder name. pole_index is unique, so folders never clash."""
    return f"pole{int(pole_index):02d}_{_geohash_suffix(pole_id)}"


# ----- rendering -----

def _render(fname: str, box_corners: dict, out_full: Path | None,
            out_crop: Path | None, highlight_box: int, pole_index: int) -> None:
    """Draw a frame with `highlight_box` in green (others gray), labeled, and save.

    Opens the image once; writes the full frame to out_full and/or a padded crop to
    out_crop (whichever are given)."""

    # Load the street image and set up PIL's drawing surface. box_corners maps
    # box_index -> (x1, y1, x2, y2) for every detection in this frame.
    img = Image.open(paths.STREET_IMAGES_DIR / fname).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(FONT_SIZE)

    # Non-Target poles first: draw every other detection in the frame as a thin gray box,
    # so you can see what the matcher chose against.
    for bi, corners in box_corners.items():
        if bi != highlight_box:
            draw.rectangle(list(corners), outline=ORPHAN_COLOR, width=ORPHAN_BOX_WIDTH)

    # The matched box, drawn last so it sits on top: thick green rectangle with a
    # "#<pole_index> -> box <box_index>" label just above its top-left corner
    # (clamped to y=0 so the label stays on-canvas for boxes at the image top).
    x1, y1, x2, y2 = box_corners[highlight_box]
    draw.rectangle([x1, y1, x2, y2], outline=MATCHED_COLOR, width=MATCHED_BOX_WIDTH)
    draw.text((x1, max(0, y1 - LABEL_ABOVE_BOX_PX)),
              f"#{int(pole_index)} -> box {int(highlight_box)}",
              fill=MATCHED_COLOR, font=font)

    # Save whichever outputs were requested. out_full is the annotated frame at
    # native resolution; out_crop is a thumbnail of the matched box plus
    # CROP_MARGIN_PX of padding, with the crop window clamped to the image edges.
    if out_full is not None:
        img.save(out_full)
    if out_crop is not None:
        w, h = img.size
        crop = img.crop((int(max(0, x1 - CROP_MARGIN_PX)), int(max(0, y1 - CROP_MARGIN_PX)),
                         int(min(w, x2 + CROP_MARGIN_PX)), int(min(h, y2 + CROP_MARGIN_PX))))
        crop.save(out_crop)


def _render_unmatched(fname: str, box_corners: dict, x_pred: float,
                      out_full: Path, pole_index: int) -> None:
    """Draw an unmatched frame: orange predicted-column line, all boxes gray."""
    img = Image.open(paths.STREET_IMAGES_DIR / fname).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(FONT_SIZE)
    for corners in box_corners.values():
        draw.rectangle(list(corners), outline=ORPHAN_COLOR, width=ORPHAN_BOX_WIDTH)
    draw.line([(x_pred, 0), (x_pred, img.height)], fill=UNMATCHED_COLOR, width=UNMATCHED_LINE_WIDTH)
    draw.text((x_pred + 6, 10), f"#{int(pole_index)}? (no box)", fill=UNMATCHED_COLOR, font=font)
    img.save(out_full)


# ----- export -----

def export_complete_and_isolated(root: Path = DEFAULT_EXPORT_DIR) -> None:
    """
    Export of both complete route and only the isolated subset of poles
    :param root:
    :return:
    """
    subset_path = run_find_isolated_poles_pipeline()
    export(root / "complete_route")
    export(root / "isolated_poles", subset_path=subset_path)


def export(out_dir: Path, matched_only: bool = False, render_images: bool = True,
           crops: bool = False, include_unmatched_renders: bool = False,
           subset_path: Path | None = None) -> None:

    matches = load_matches()
    dets = load_detections()

    # Restrict to a subset of poles (e.g. the isolated ones from
    # find_isolated_poles) — a CSV with (image_filename, pole_id). Same subset
    # semantics as evaluate.compare/sweep, so the exported set matches what was scored.
    if subset_path is not None:
        subset = pd.read_csv(subset_path)[["image_filename", "pole_id"]].drop_duplicates()
        n0 = len(matches)
        matches = matches.merge(subset, on=["image_filename", "pole_id"], how="inner")
        print(f"subset: exporting {len(matches)}/{n0} pole-frames from {subset_path.name}")

    # Corners looked up per frame on demand (a frame is shared by several poles).
    corners_cache: dict[str, dict] = {}

    def frame_corners(fname: str) -> dict:
        if fname not in corners_cache:
            corners_cache[fname] = _box_corners_for_frame(dets, fname)
        return corners_cache[fname]

    by_pole = out_dir / "by_pole"
    out_dir.mkdir(parents=True, exist_ok=True)
    if render_images:
        # Clear stale renders so an old matching's images never linger.
        if by_pole.exists():
            shutil.rmtree(by_pole)
        by_pole.mkdir(parents=True)

    # All of a pole's rows contiguous, in route order.
    matches = matches.sort_values(["pole_id", "frame_number"]).reset_index(drop=True)

    detail_rows = []
    n_rendered = 0
    for pole_id, grp in matches.groupby("pole_id", sort=True):
        pole_index = int(grp.pole_index.iloc[0])
        folder = by_pole / _pole_folder_name(pole_index, pole_id)

        for r in grp.itertuples():
            is_matched = (r.status == "matched") and pd.notna(r.box_index)
            rec = {
                "pole_id": pole_id,
                "pole_index": pole_index,
                "frame_number": int(r.frame_number),
                "image_filename": r.image_filename,
                "box_index": int(r.box_index) if is_matched else "",
                "residual_px": round(float(r.residual_px), 1) if is_matched else "",
                "status": r.status,
                "rendered_path": "",
                "batch": r.batch,
            }
            for c in BBOX_COLS:
                rec[c] = ""

            if is_matched:
                corners = frame_corners(r.image_filename)
                box = int(r.box_index)
                if box in corners:
                    rec["bbox_x1"], rec["bbox_y1"], rec["bbox_x2"], rec["bbox_y2"] = corners[box]
                    if render_images:
                        folder.mkdir(parents=True, exist_ok=True)
                        stem = f"frame{int(r.frame_number):03d}_box{box}"
                        full = folder / f"{stem}.jpg"
                        crop = folder / f"{stem}_crop.jpg" if crops else None
                        _render(r.image_filename, corners, full, crop, box, pole_index)
                        rec["rendered_path"] = str(full.relative_to(out_dir))
                        n_rendered += 1
            elif render_images and include_unmatched_renders:
                corners = frame_corners(r.image_filename)
                folder.mkdir(parents=True, exist_ok=True)
                full = folder / f"frame{int(r.frame_number):03d}_nobox.jpg"
                _render_unmatched(r.image_filename, corners, float(r.x_pred), full, pole_index)
                rec["rendered_path"] = str(full.relative_to(out_dir))

            if is_matched or not matched_only:
                detail_rows.append(rec)

    # ----- write CSVs -----
    col_order = ["pole_id", "pole_index", "frame_number", "image_filename",
                 "box_index", *BBOX_COLS, "residual_px", "status", "rendered_path", "batch"]
    detail = pd.DataFrame(detail_rows)[col_order]
    detail.to_csv(out_dir / "poles_index.csv", index=False)

    summary = (
        matches.assign(_m=(matches.status == "matched"))
        .groupby("pole_id")
        .apply(lambda g: pd.Series({
            "pole_index": int(g.pole_index.iloc[0]),
            "n_frames_observed": len(g),
            "n_matched": int(g._m.sum()),
            "first_frame": int(g.frame_number.min()),
            "last_frame": int(g.frame_number.max()),
            "median_residual_px": round(float(g.loc[g._m, "residual_px"].median()), 1) if g._m.any() else "",
            "batch": g.batch.iloc[0],
        }), include_groups=False)
        .reset_index()
        .sort_values("pole_index")
    )
    summary.to_csv(out_dir / "poles_summary.csv", index=False)

    # ----- report -----
    print(f"exported {detail.pole_id.nunique()} poles, {len(detail)} pole-image rows")
    print(f"  poles_index.csv   ({len(detail)} rows)")
    print(f"  poles_summary.csv ({len(summary)} poles)")
    if render_images:
        print(f"  by_pole/          ({n_rendered} images rendered{' + crops' if crops else ''})")
    print(f"  -> {out_dir}")


def run_export_poles_pipeline():
    p = argparse.ArgumentParser(description="Export a pole-centric CSV + per-pole image folders.")
    p.add_argument("--out", type=Path, default=DEFAULT_EXPORT_DIR)
    p.add_argument("--matched-only", action="store_true",
                   help="CSV: drop frames where the pole wasn't matched")
    p.add_argument("--no-images", action="store_true", help="write CSVs only, skip rendering")
    p.add_argument("--crops", action="store_true",
                   help="also save a tight crop around each box")
    p.add_argument("--include-unmatched-renders", action="store_true",
                   help="also render unmatched frames (predicted line, no box)")
    p.add_argument("--subset", type=Path, default=None,
                   help="CSV of (image_filename, pole_id) to export only those poles "
                        "(e.g. isolated_poles.csv from find_isolated_poles)")
    args = p.parse_args()

    export(args.out, matched_only=args.matched_only, render_images=not args.no_images,
           crops=args.crops, include_unmatched_renders=args.include_unmatched_renders,
           subset_path=args.subset)

def main():
    run_export_poles_pipeline()

if __name__ == "__main__":
    main()
