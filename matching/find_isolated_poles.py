"""Find spatially-isolated poles and evaluate the matcher on just them.

Produces a subset CSV of (image_filename, pole_id) and hands it to the STANDARD
evaluation (`evaluate.compare` / `evaluate.sweep`) via their --subset support, so
no scoring logic is duplicated here — this only decides *which* poles to score.

"Isolated" = alone in its frame (the frame contains exactly one observed pole).
This is the subset shown to match far better; see
documentation/matching_evaluation_findings.md.

Run:
    python3 -m matching.find_isolated_poles                 # write subset + compare
    python3 -m matching.find_isolated_poles --contrast      # also score the crowded rest
    python3 -m matching.find_isolated_poles --sweep         # also gate-sweep the subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths  # noqa: E402
from matching.evaluate import compare, sweep  # noqa: E402


def isolated_poles(ground_truth_path: Path) -> pd.DataFrame:
    """Return the (image_filename, pole_id) of poles alone in their frame."""
    truth = pd.read_csv(ground_truth_path)
    per_frame = truth.groupby("image_filename").size()
    alone_frames = per_frame[per_frame == 1].index
    return truth[truth.image_filename.isin(alone_frames)][["image_filename", "pole_id"]]


def main() -> int:
    p = argparse.ArgumentParser(description="Find isolated poles and score the matcher on them.")
    p.add_argument("ground_truth", type=Path, nargs="?",
                   default=paths.MATCHING_DIR / "ground_truth.csv")
    p.add_argument("--out", type=Path, default=paths.MATCHING_DIR / "isolated_poles.csv",
                   help="where to write the subset CSV")
    p.add_argument("--contrast", action="store_true",
                   help="also score the complement (poles that share a frame)")
    p.add_argument("--sweep", action="store_true",
                   help="also run the GATE_PX sweep on the isolated subset")
    args = p.parse_args()

    truth = pd.read_csv(args.ground_truth)
    iso = isolated_poles(args.ground_truth)
    iso.to_csv(args.out, index=False)
    print(f"{len(iso)}/{len(truth)} poles are isolated (alone in frame) -> {args.out}\n")

    print("=== isolated (alone in frame) ===")
    compare(args.ground_truth, subset_path=args.out)

    if args.contrast:
        rest = truth.merge(iso.assign(_iso=1), on=["image_filename", "pole_id"], how="left")
        rest_path = args.out.with_name(args.out.stem + "_complement.csv")
        rest[rest._iso.isna()][["image_filename", "pole_id"]].to_csv(rest_path, index=False)
        print("\n=== crowded (shares frame) — for contrast ===")
        compare(args.ground_truth, subset_path=rest_path)

    if args.sweep:
        print("\n=== gate sweep on the isolated subset ===")
        sweep(args.ground_truth, subset_path=args.out, target_precision=0.8)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
