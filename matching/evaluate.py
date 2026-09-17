"""Evaluate the matcher against a one-time hand annotation.

Workflow (annotate once, compare many times):

    1. python3 -m matching.evaluate --batch NAME template
         -> writes <route>/evaluation/ground_truth_template.csv, one row per
            (frame, pole) with an empty `true_box_index` column. Copy it to
            ground_truth.csv (same directory) and fill in the correct box_index for each pole (leave
            blank if no box is correct / the pole isn't visible). Use the
            box-index-labeled images from `visualize_matches --label-boxes` to
            read off box numbers.

    2. Tune GATE_PX / THETA_MAX / MIN_SCORE, re-run matching, then:
       python3 -m matching.evaluate --batch NAME compare
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
from config.paths import RoutePaths  # noqa: E402
from localization.localization_pipeline import load_observations  # noqa: E402
from matching.matching_pipeline import load_matches, match_all  # noqa: E402

# GATE_PX sweep resolution; the range runs 0..(max observed residual).
SWEEP_STEP_PX = 25.0


def write_template(route: RoutePaths) -> Path:
    """Write one row per (frame, pole) with an empty true_box_index to fill in.

    Sourced from observations (not from the matches), so the template is
    independent of the matching parameters — the poles you need to judge don't
    change when you tune the algorithm.

    Returns the template path (<route>/evaluation/ground_truth_template.csv).
    """
    template_path = route.ground_truth_template
    obs = load_observations(route)
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

    template_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(template_path, index=False)
    print(f"wrote {len(template)} rows to {template_path}")
    print("Copy it to ground_truth.csv and fill in true_box_index (+ pole_visible on")
    print("blank-box rows), then run:")
    print(f"  python3 -m matching.evaluate --batch {route.batch!r} compare")
    return template_path


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


def _read_truth(ground_truth_path: Path, subset_path: Path | None = None) -> pd.DataFrame:
    """Load a ground-truth CSV (numeric true_box_index, boolean _visible), optionally
    restricted to a subset of poles.

    subset_path is a CSV with at least (image_filename, pole_id); only those poles are
    kept. This is the one place subset filtering happens, so compare/sweep and any
    caller (e.g. find_isolated_poles) share identical semantics."""
    truth = pd.read_csv(ground_truth_path)
    # Blank cells read as NaN; coerce the box column to numeric.
    truth["true_box_index"] = pd.to_numeric(truth["true_box_index"], errors="coerce")
    # pole_visible is optional (older sheets predate it). Absent -> treat all as
    # not-flagged, which collapses detector_miss back into correct_reject.
    if "pole_visible" in truth.columns:
        truth["_visible"] = pd.to_numeric(truth["pole_visible"], errors="coerce").fillna(0) == 1
    else:
        truth["_visible"] = False

    if subset_path is not None:
        subset = pd.read_csv(subset_path)[["image_filename", "pole_id"]].drop_duplicates()
        n0 = len(truth)
        truth = truth.merge(subset, on=["image_filename", "pole_id"], how="inner")
        print(f"subset: scoring {len(truth)}/{n0} poles from {subset_path.name}\n")
    return truth


def compare(route: RoutePaths, ground_truth_path: Path | None = None,
            subset_path: Path | None = None, save_path: Path | None = None) -> pd.DataFrame:
    """Score one route's current matches against a filled-in ground-truth CSV.

    ground_truth_path=None -> the route's own ground_truth.csv.

    Prints a report of the useful statistics; with save_path, writes the same
    report to a .txt file. The report covers: the category breakdown, the matching
    metrics (precision / recall / F1 / matcher accuracy / detector recall), and a
    little match geometry + error diagnostics (median residual, the wrong_box angle,
    and why the misses missed).
    """
    if ground_truth_path is None:
        ground_truth_path = route.ground_truth
    truth = _read_truth(ground_truth_path, subset_path)
    visible = truth["_visible"]

    # Pull residual/theta/status too (already in load_matches) so the report can
    # include geometry + diagnostics, not just the box decision.
    algo = load_matches(route)[["image_filename", "pole_id", "pole_index", "box_index",
                           "residual_px", "theta_deg", "status"]]

    # Left-join so every annotated pole is scored, matched or not.
    merged = truth.merge(algo, on=["image_filename", "pole_id"], how="left",
                         suffixes=("", "_algo"))
    merged["category"] = [
        _category(t, a, v)
        for t, a, v in zip(merged.true_box_index, merged.box_index, visible)
    ]

    counts = merged["category"].value_counts()
    total = len(merged)
    cm = counts.get("correct_match", 0)
    wrong = counts.get("wrong_box", 0)
    false = counts.get("false_match", 0)
    missed = counts.get("missed", 0)
    reject = counts.get("correct_reject", 0)
    det_miss = counts.get("detector_miss", 0)
    p, r, f1 = _prf(counts)
    matcher_total = total - det_miss
    detected = cm + wrong + missed
    denom_det = detected + det_miss

    def px(x):
        return "n/a" if x is None or pd.isna(x) else f"{float(x):.1f}"

    # ----- Build the report once, so print and file are identical -----
    lines = [f"scored {total} annotated poles"
             + (f"  (subset: {subset_path.name})" if subset_path is not None else "")]

    lines.append("\ncategory breakdown:")
    for cat in ["correct_match", "correct_reject", "detector_miss",
                "wrong_box", "missed", "false_match"]:
        n = counts.get(cat, 0)
        lines.append(f"  {cat:15s} {n:4d}  ({n / total:.0%})")

    lines.append("\nmatching metrics:")
    lines.append(f"  match precision:  {p:.1%}")
    lines.append(f"  match recall:     {r:.1%}")
    lines.append(f"  match F1:         {f1:.1%}")
    lines.append(f"  matcher accuracy: {(cm + reject) / max(matcher_total, 1):.1%}"
                 f"  (excl. {det_miss} detector misses)")
    lines.append(f"  detector recall:  {detected / max(denom_det, 1):.1%}"
                 f"  ({detected}/{denom_det} visible poles boxed)")

    # Match geometry + error diagnostics (from the residual/theta/status columns).
    matched = merged[merged.status == "matched"]
    wrongbox = merged[merged.category == "wrong_box"]
    missed_split = merged.loc[merged.category == "missed", "status"].value_counts()
    lines.append("\ngeometry & diagnostics:")
    lines.append(f"  median residual (matched):  {px(matched.residual_px.median())} px")
    lines.append(f"  wrong_box median |theta|:   {px(wrongbox.theta_deg.abs().median())} deg")
    split = ", ".join(f"{k} {int(v)}" for k, v in missed_split.items()) or "none"
    lines.append(f"  missed breakdown:           {split}")

    report = "\n".join(lines)
    print(report)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report + "\n")
        print(f"\nsaved report -> {save_path}")

    return merged


def _prf(counts) -> tuple[float, float, float]:
    """Precision, recall, F1 on the matcher's 'there is a box' decision, from a
    dict of category -> count."""
    cm = counts.get("correct_match", 0)
    wrong = counts.get("wrong_box", 0)
    false = counts.get("false_match", 0)
    missed = counts.get("missed", 0)
    denom_p = cm + wrong + false
    denom_r = cm + wrong + missed
    p = cm / denom_p if denom_p else 0.0
    r = cm / denom_r if denom_r else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def sweep(route: RoutePaths, ground_truth_path: Path | None = None,
          subset_path: Path | None = None) -> pd.DataFrame:
    """Precision/recall/F1 as a function of GATE_PX, from one route's annotation.

    ground_truth_path=None -> the route's own ground_truth.csv.

    Runs the matcher ONCE with the gate disabled (gate_px=inf) so every pole's
    assigned box + residual is known, then re-gates in pandas every SWEEP_STEP_PX
    from 0 to the max observed residual. THETA_MAX and MIN_SCORE are held at
    their current module values (they change the assignment and would need a
    re-run per value — see the design doc).
    """
    if ground_truth_path is None:
        ground_truth_path = route.ground_truth
    truth = _read_truth(ground_truth_path, subset_path)
    visible = truth["_visible"]

    # Ungated run: with gate_px=inf every assignable cone pole is 'matched', so we
    # recover its assigned box_index and residual before any gating.
    ungated = match_all(route, gate_px=float("inf"))
    assigned = ungated[ungated.status == "matched"][
        ["image_filename", "pole_id", "box_index", "residual_px"]
    ]
    merged = truth.merge(assigned, on=["image_filename", "pole_id"], how="left")

    resid = merged["residual_px"]
    assigned_box = merged["box_index"]
    true_box = merged["true_box_index"]

    max_resid = float(resid.max()) if resid.notna().any() else 0.0
    n_gates = int(max_resid // SWEEP_STEP_PX) + 2
    gates = [round(i * SWEEP_STEP_PX, 3) for i in range(n_gates)]

    records = []
    for g in gates:
        # The matcher emits the assigned box only if it's within the gate; an
        # unassigned pole (residual NaN) stays no-box at every gate.
        algo_box = assigned_box.where(resid <= g)
        cats = [_category(t, a, v) for t, a, v in zip(true_box, algo_box, visible)]
        c = pd.Series(cats).value_counts().to_dict()
        p, r, f1 = _prf(c)
        records.append({
            "gate_px": g,
            "correct_match": c.get("correct_match", 0),
            "wrong_box": c.get("wrong_box", 0),
            "missed": c.get("missed", 0),
            "false_match": c.get("false_match", 0),
            "precision": p, "recall": r, "f1": f1,
        })
    table = pd.DataFrame(records)

    # ----- Report -----
    print(f"swept GATE_PX {gates[0]:.0f}..{gates[-1]:.0f} step {SWEEP_STEP_PX:.0f} "
          f"over {len(truth)} annotated poles "
          f"(THETA_MAX / MIN_SCORE held at current values)\n")
    print(f"  {'gate':>6}  {'cm':>4} {'wrong':>5} {'miss':>4} {'false':>5}   "
          f"{'prec':>6} {'recall':>6} {'F1':>6}")
    for r in records:
        print(f"  {r['gate_px']:6.0f}  {r['correct_match']:4d} {r['wrong_box']:5d} "
              f"{r['missed']:4d} {r['false_match']:5d}   "
              f"{r['precision']:6.1%} {r['recall']:6.1%} {r['f1']:6.1%}")

    best = table.loc[table.f1.idxmax()]
    print(f"\n  best F1: gate={best.gate_px:.0f}  "
          f"(prec {best.precision:.1%}, recall {best.recall:.1%}, F1 {best.f1:.1%})")
    return table


def run_evaluation_pipeline(route: RoutePaths) -> bool:
    """Pipeline step: evaluate one route's matches if it has been annotated.

    Always (re)writes the annotation template into the route's evaluation/ dir.
    If a filled-in ground_truth.csv sits next to it, scores the current matches
    and saves full_report.txt (and isolated_report.txt when the route's isolated
    subset CSV exists) under evaluation/reports/. Otherwise prints a one-line
    skip: a missing ground truth is the normal state on a route's first run.

    Returns True if scoring ran.
    """
    write_template(route)
    if not route.ground_truth.exists():
        print(f"\nevaluation skipped: no ground truth at {route.ground_truth}")
        print(f"  fill in {route.ground_truth_template.name} and save it as "
              f"{route.ground_truth.name} in the same dir to score this route")
        return False

    print("\n=== evaluation: all annotated poles ===")
    compare(route, save_path=route.eval_reports_dir / "full_report.txt")
    if route.isolated_poles.exists():
        print("\n=== evaluation: isolated subset (pole alone in its frame) ===")
        compare(route, subset_path=route.isolated_poles,
                save_path=route.eval_reports_dir / "isolated_report.txt")
    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate matcher vs hand annotation.")
    paths.add_batch_arg(parser)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template", help="write the annotation template")
    scoring_cmds = {}
    for name, help_text in [("compare", "score current matches against filled ground truth"),
                            ("sweep", "precision/recall/F1 vs GATE_PX from one annotation")]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("ground_truth", type=Path, nargs="?", default=None,
                         help="filled-in ground-truth CSV (default: <route>/evaluation/ground_truth.csv)")
        cmd.add_argument("--subset", type=Path, default=None,
                         help="CSV of (image_filename, pole_id) to score only those poles")
        scoring_cmds[name] = cmd
    scoring_cmds["compare"].add_argument("--save", type=Path, default=None, metavar="FILE.txt",
                                         help="also write the report to this text file")
    args = parser.parse_args()

    route = paths.route_paths(args.batch)
    if args.command == "template":
        write_template(route)
    elif args.command == "compare":
        compare(route, args.ground_truth, subset_path=args.subset, save_path=args.save)
    else:
        sweep(route, args.ground_truth, subset_path=args.subset)


if __name__ == "__main__":
    main()
