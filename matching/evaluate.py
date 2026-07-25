"""Evaluate the matcher against a one-time hand annotation.

Workflow (annotate once, compare many times):

    1. python3 -m matching.evaluate template
         -> writes ground_truth_template.csv, one row per (frame, pole) with an
            empty `true_box_index` column. Copy it to ground_truth.csv and fill
            in the correct box_index for each pole (leave blank if no box is
            correct / the pole isn't visible). Use the box-index-labeled images
            from `visualize_matches --label-boxes` to read off box numbers.

    2. Tune GATE_PX / THETA_MAX / MIN_SCORE, re-run matching, then:
       python3 -m matching.evaluate compare ground_truth.csv
         -> scores the current matches against your annotation. Re-run freely;
            your annotation never has to change.

Ground truth is keyed on (image_filename, pole_id) -> true_box_index. box_index
is assigned at detection time, so it is stable across matching-parameter changes
(only re-running detection would invalidate the annotation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths  # noqa: E402
from localization.localization_pipeline import load_observations  # noqa: E402
from matching.matching_pipeline import load_matches  # noqa: E402

TEMPLATE_PATH = paths.MATCHING_DIR / "ground_truth_template.csv"


def write_template() -> None:
    """Write one row per (frame, pole) with an empty true_box_index to fill in.

    Sourced from observations (not from the matches), so the template is
    independent of the matching parameters — the poles you need to judge don't
    change when you tune the algorithm.
    """
    obs = load_observations()
    template = (
        obs[["frame_number", "image_filename", "pole_index", "pole_id"]]
        .drop_duplicates(["image_filename", "pole_id"])
        # Frame-major order: step through frames along the route; a frame's poles
        # sit together. pole_index is already sequential along the route, so the
        # Nth pole to appear as you step is pole_index N.
        .sort_values(["frame_number", "pole_index"])
        .reset_index(drop=True)
    )
    template["true_box_index"] = ""  # <- you fill this in (blank = no correct box)
    # Only fill on rows you leave true_box_index BLANK: 1 = pole is visible but has
    # no box (a detector miss), blank = pole not visible (a fair no-box). Rows that
    # HAVE a box are visible by definition, so leave pole_visible blank there. This
    # is what lets scoring separate detector misses from matcher misses.
    template["pole_visible"] = ""

    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(TEMPLATE_PATH, index=False)
    print(f"wrote {len(template)} rows to {TEMPLATE_PATH}")
    print("Copy it to ground_truth.csv and fill in true_box_index (+ pole_visible on")
    print("blank-box rows), then run:")
    print("  python3 -m matching.evaluate compare ground_truth.csv")


def _category(true_box, algo_box, pole_visible) -> str:
    """Classify one (frame, pole) by comparing the true box to the algorithm's.

    pole_visible only disambiguates the no-true-box cases: True means the pole was
    visible but has no box (the detector missed it), so a matcher "no box" is not
    really a matcher success — it's charged to detection instead.
    """
    true_has = pd.notna(true_box)
    algo_has = pd.notna(algo_box)
    if true_has and algo_has:
        return "correct_match" if int(true_box) == int(algo_box) else "wrong_box"
    if true_has and not algo_has:
        return "missed"          # a box was correct but the algorithm found none
    if not true_has and algo_has:
        return "false_match"     # algorithm matched a box that shouldn't be there
    # Neither has a box. Split by why the pole has no correct box:
    if pole_visible:
        return "detector_miss"   # pole was visible but the detector drew no box
    return "correct_reject"      # pole not visible; both agree there's no box


def compare(ground_truth_path: Path) -> pd.DataFrame:
    """Score the current matches against a filled-in ground-truth CSV."""
    truth = pd.read_csv(ground_truth_path)
    # Blank cells read as NaN; coerce the box column to numeric.
    truth["true_box_index"] = pd.to_numeric(truth["true_box_index"], errors="coerce")
    # pole_visible is optional (older sheets predate it). Absent -> treat all as
    # not-flagged, which collapses detector_miss back into correct_reject.
    if "pole_visible" in truth.columns:
        visible = pd.to_numeric(truth["pole_visible"], errors="coerce").fillna(0) == 1
    else:
        visible = pd.Series(False, index=truth.index)

    algo = load_matches()[["image_filename", "pole_id", "pole_index", "box_index"]]

    # Left-join so every annotated pole is scored, matched or not.
    merged = truth.merge(algo, on=["image_filename", "pole_id"], how="left",
                         suffixes=("", "_algo"))
    merged["category"] = [
        _category(t, a, v)
        for t, a, v in zip(merged.true_box_index, merged.box_index, visible)
    ]

    # ----- Report -----
    counts = merged["category"].value_counts()
    total = len(merged)

    print(f"scored {total} annotated poles\n")
    for cat in ["correct_match", "correct_reject", "detector_miss",
                "wrong_box", "missed", "false_match"]:
        n = counts.get(cat, 0)
        print(f"  {cat:15s} {n:4d}  ({n / total:.0%})")

    cm = counts.get("correct_match", 0)
    wrong = counts.get("wrong_box", 0)
    false = counts.get("false_match", 0)
    missed = counts.get("missed", 0)
    reject = counts.get("correct_reject", 0)
    det_miss = counts.get("detector_miss", 0)

    # Matcher accuracy excludes detector_miss: with no box to work from, the
    # matcher's "no box" is neither a win nor a loss for it.
    matcher_total = total - det_miss
    if matcher_total:
        print(f"\n  matcher accuracy: {(cm + reject) / matcher_total:.1%}"
              f"  (excl. {det_miss} detector misses)")

    # Precision/recall on the matcher's "there is a box" decision.
    denom_p = cm + wrong + false
    denom_r = cm + wrong + missed
    if denom_p:
        print(f"  match precision:  {cm / denom_p:.1%}")
    if denom_r:
        print(f"  match recall:     {cm / denom_r:.1%}")

    # Detection recall: of poles you could see, how many the detector boxed.
    # Visible poles = those with a true box (detected) + detector_miss (missed).
    detected = cm + wrong + missed
    denom_det = detected + det_miss
    if denom_det:
        print(f"  detector recall:  {detected / denom_det:.1%}"
              f"  ({detected}/{denom_det} visible poles boxed)")

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate matcher vs hand annotation.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template", help="write the annotation template")
    cmp = sub.add_parser("compare", help="score current matches against filled ground truth")
    cmp.add_argument("ground_truth", type=Path, help="your filled-in ground_truth.csv")
    args = parser.parse_args()

    if args.command == "template":
        write_template()
    elif args.command == "compare":
        compare(args.ground_truth)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
