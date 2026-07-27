"""Find spatially-isolated poles (alone in their frame).

"Isolated" = alone in its frame (the frame contains exactly one observed pole) —
the subset shown to match far better; see
documentation/matching_evaluation_findings.md.

Importable helper, not a script. Typical use:

    from matching.find_isolated_poles import isolated_poles

    iso = isolated_poles(ground_truth_path)     # DataFrame of (image_filename, pole_id)
    iso.to_csv(subset_path, index=False)        # persist, then pass subset_path to
    #   evaluate.compare / evaluate.sweep / export_poles.export, which all take a subset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def isolated_poles(ground_truth_path: Path) -> pd.DataFrame:
    """Return the (image_filename, pole_id) of poles alone in their frame."""
    truth = pd.read_csv(ground_truth_path)
    per_frame = truth.groupby("image_filename").size()
    alone_frames = per_frame[per_frame == 1].index
    return truth[truth.image_filename.isin(alone_frames)][["image_filename", "pole_id"]]
