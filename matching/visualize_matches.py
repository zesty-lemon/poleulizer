"""Draw a frame's pole<->box matches onto the image so they can be eyeballed.

For one street-image frame this overlays three things:

    green box + "#N"     a matched pole (N is its pole_index) on its detected box
    orange line + "#N?"  an UNMATCHED pole, drawn at the column where it was
                         predicted to appear (so you can see what was missed)
    gray box             an orphan detection: a box no pole claimed
                         (a false positive, or a left-of-road / unlabeled pole)

It only reads the matches + detection tables and the image file — no model, no
GPU. Useful for sanity-checking the matcher frame by frame.

Run:
    python3 -m matching.visualize_matches                     # busiest matched frame
    python3 -m matching.visualize_matches --frame <name>      # one specific frame
    python3 -m matching.visualize_matches --save /tmp/out.jpg # save instead of opening
    python3 -m matching.visualize_matches --all <dir>         # render every frame into dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths  # noqa: E402
from detection.run_detection import load_detections  # noqa: E402
from matching.matching_pipeline import load_matches  # noqa: E402

# ----- Drawing style -----
# One place to tweak how the overlay looks. Images are ~3840px wide, so lines
# and fonts need to be chunky to be visible.
MATCHED_COLOR = "lime"
UNMATCHED_COLOR = "orange"
ORPHAN_COLOR = "gray"

MATCHED_BOX_WIDTH = 8
ORPHAN_BOX_WIDTH = 4
UNMATCHED_LINE_WIDTH = 3

FONT_SIZE = 64
LABEL_ABOVE_BOX_PX = 70  # how far above a box's top edge to place its "#N" label


def _load_font(size: int):
    """Return a large legible TrueType font, falling back to PIL's tiny default.

    We try a couple of standard macOS font paths; if none load (e.g. on another
    OS) we use the built-in bitmap font so the script still runs.
    """
    for font_path in ("/System/Library/Fonts/Helvetica.ttc", "/Library/Fonts/Arial.ttf"):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _box_corners_for_frame(boxes, fname: str) -> dict[int, tuple]:
    """Map box_index -> (x1, y1, x2, y2) for the real detections in one frame.

    The matches table only stores box_index, not the pixel corners, so we look
    the corners back up in the detections table here. Sentinel rows (box_index
    == -1, meaning "frame had zero detections") are skipped.
    """
    frame_boxes = boxes[(boxes.image_filename == fname) & (boxes.box_index >= 0)]
    return {
        int(box.box_index): (box.bbox_x1, box.bbox_y1, box.bbox_x2, box.bbox_y2)
        for box in frame_boxes.itertuples()
    }


def _draw_matched_pole(draw, font, pole, corners) -> None:
    """Green box around the matched detection, labeled with the pole's #index."""
    x1, y1, x2, y2 = corners
    draw.rectangle([x1, y1, x2, y2], outline=MATCHED_COLOR, width=MATCHED_BOX_WIDTH)
    label_y = max(0, y1 - LABEL_ABOVE_BOX_PX)  # clamp so the label stays on-image
    draw.text((x1, label_y), f"#{pole.pole_index}", fill=MATCHED_COLOR, font=font)


def _draw_unmatched_pole(draw, font, pole, image_height) -> None:
    """Orange vertical line at the pole's PREDICTED column (it got no box).

    Seeing where the prediction landed tells you whether the detector simply
    missed a pole that's clearly there, or the pole genuinely isn't visible.
    """
    x = pole.x_pred
    draw.line([(x, 0), (x, image_height)], fill=UNMATCHED_COLOR, width=UNMATCHED_LINE_WIDTH)
    draw.text((x + 6, 10), f"#{pole.pole_index}?", fill=UNMATCHED_COLOR, font=font)


def _draw_orphan_boxes(draw, box_corners, matched_box_ids) -> int:
    """Gray box around every detection no pole claimed. Returns how many there were."""
    orphan_count = 0
    for box_index, corners in box_corners.items():
        if box_index not in matched_box_ids:
            draw.rectangle(list(corners), outline=ORPHAN_COLOR, width=ORPHAN_BOX_WIDTH)
            orphan_count += 1
    return orphan_count


def visualize_frame(fname: str, save: Path | None = None,
                    matches=None, boxes=None) -> None:
    """Render the overlay for one frame, then either save it or open it.

    matches/boxes can be passed in so batch mode loads them once instead of
    re-reading the parquet files for every frame.
    """
    # ----- 1. Gather this frame's data -----
    if matches is None:
        matches = load_matches()
    if boxes is None:
        boxes = load_detections()

    matches_for_frame = matches[matches.image_filename == fname]
    if matches_for_frame.empty:
        raise SystemExit(f"no match rows for frame {fname!r}")

    box_corners = _box_corners_for_frame(boxes, fname)

    # ----- 2. Open the image and set up drawing -----
    img = Image.open(paths.STREET_IMAGES_DIR / fname).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(FONT_SIZE)

    # ----- 3. Draw each pole (matched -> box+label, unmatched -> predicted line) -----
    matched_box_ids = set()  # remember which boxes got claimed, so we can find orphans
    for pole in matches_for_frame.itertuples():
        if pole.status == "matched":
            matched_box_ids.add(int(pole.box_index))
            _draw_matched_pole(draw, font, pole, box_corners[int(pole.box_index)])
        else:
            _draw_unmatched_pole(draw, font, pole, img.height)

    # ----- 4. Draw the leftover (orphan) boxes -----
    orphan_count = _draw_orphan_boxes(draw, box_corners, matched_box_ids)

    matched_count = (matches_for_frame.status == "matched").sum()
    print(f"{fname}: {matched_count} matched, {orphan_count} orphan box(es)")

    # ----- 5. Save to disk, or pop it open in the OS image viewer -----
    if save:
        img.save(save)
        print(f"saved {save}")
    else:
        img.show()


def label_boxes_frame(fname: str, save: Path | None = None, boxes=None) -> None:
    """Draw every detection box labeled with its box_index (annotation aid).

    Use this to fill in the ground-truth template: look at a frame, read off
    which cyan box number sits on each pole, and record it as that pole's
    true_box_index. Shows the box NUMBER, deliberately not the algorithm's guess.
    """
    if boxes is None:
        boxes = load_detections()

    box_corners = _box_corners_for_frame(boxes, fname)
    img = Image.open(paths.STREET_IMAGES_DIR / fname).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(FONT_SIZE)

    for box_index, (x1, y1, x2, y2) in box_corners.items():
        draw.rectangle([x1, y1, x2, y2], outline="cyan", width=6)
        draw.text((x1, max(0, y1 - LABEL_ABOVE_BOX_PX)), f"{box_index}",
                  fill="cyan", font=font)

    print(f"{fname}: {len(box_corners)} boxes labeled")
    if save:
        img.save(save)
        print(f"saved {save}")
    else:
        img.show()


def _busiest_matched_frame(matches) -> str:
    """The frame with the most matched poles — a good default to eyeball first."""
    matched = matches[matches.status == "matched"]
    return matched.groupby("image_filename").size().idxmax()


def _render_all(out_dir: Path, label_boxes: bool = False) -> None:
    """Render every frame into out_dir (loads data once).

    label_boxes=True renders the box-index annotation aid instead of the match
    overlay.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    boxes = load_detections()

    frames = sorted(matches.image_filename.unique())
    print(f"rendering {len(frames)} frames -> {out_dir}")
    for i, fname in enumerate(frames, start=1):
        # Frame filenames already end in .jpg, so reuse them as output names.
        if label_boxes:
            label_boxes_frame(fname, out_dir / fname, boxes=boxes)
        else:
            visualize_frame(fname, out_dir / fname, matches=matches, boxes=boxes)
        if i % 50 == 0:
            print(f"  {i}/{len(frames)}")
    print("done")


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize pole<->box matches for a frame.")
    parser.add_argument("--frame", default=None,
                        help="image_filename to render (default: busiest matched frame)")
    parser.add_argument("--save", type=Path, default=None,
                        help="save to this path instead of opening a viewer")
    parser.add_argument("--all", type=Path, default=None, metavar="DIR",
                        help="render every frame in the matches table into DIR")
    parser.add_argument("--label-boxes", action="store_true",
                        help="annotation aid: label each box by box_index instead of "
                             "showing the match overlay")
    args = parser.parse_args()

    # Batch mode: render everything and exit.
    if args.all is not None:
        _render_all(args.all, label_boxes=args.label_boxes)
        return 0

    # Single-frame mode: default to the busiest matched frame if none given.
    fname = args.frame
    if fname is None:
        fname = _busiest_matched_frame(load_matches())
        print(f"no --frame given; using busiest matched frame: {fname}")

    if args.label_boxes:
        label_boxes_frame(fname, args.save)
    else:
        visualize_frame(fname, args.save)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
