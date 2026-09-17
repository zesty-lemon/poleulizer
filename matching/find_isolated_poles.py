"""Find spatially-isolated poles (alone in their frame) for the current route.

"Isolated" = alone in its frame: the frame contains exactly one localized pole.
This is the subset shown to match better

Sourced from the localization observations, not from a ground-truth annotation,
so it can be computed for any onboarded route with no hand labeling.

The subset is at pole-FRAME granularity: a pole is usually alone in some frames
and shares later frames with its neighbours, so the result keeps only the frames
where it is alone. Very few poles are isolated in every frame they appear in.

Output is (image_filename, pole_id), the subset format accepted by
evaluate.compare / evaluate.sweep / export_poles.export via `subset_path`.

Run:
    python3 -m matching.find_isolated_poles              # write ../data/matching/isolated_poles.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths  # noqa: E402
from localization.localization_pipeline import load_observations  # noqa: E402

SUBSET_COLS = ["image_filename", "pole_id"]


def isolated_poles(obs: pd.DataFrame | None = None, batch: str | None = None) -> pd.DataFrame:
    """Return the (image_filename, pole_id) of poles alone in their frame.

    obs defaults to the localization observations on disk. batch optionally
    restricts to one route before counting, so a frame is only judged against
    the poles of its own route.
    """
    if obs is None:
        obs = load_observations()
    if batch is not None:
        obs = obs[obs.batch == batch]
    obs = obs.drop_duplicates(SUBSET_COLS)

    # Keep only frames with exactly one pole in them.
    per_frame = obs.groupby("image_filename").pole_id.nunique()
    alone_frames = per_frame[per_frame == 1].index
    return obs[obs.image_filename.isin(alone_frames)][SUBSET_COLS].reset_index(drop=True)


def run_find_isolated_poles_pipeline() -> Path:
    """Compute the isolated subset from the observations on disk and persist it
    to paths.ISOLATED_POLES_PATH; returns that path."""
    obs = load_observations()
    iso = isolated_poles(obs)
    out_path = paths.ISOLATED_POLES_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    iso.to_csv(out_path, index=False)
    n_total = len(obs.drop_duplicates(SUBSET_COLS))
    print(f"isolated poles: {len(iso)}/{n_total} pole-frames "
          f"({iso.pole_id.nunique()}/{obs.pole_id.nunique()} poles) -> {out_path}")
    return out_path


def main():
    run_find_isolated_poles_pipeline()


if __name__ == "__main__":
    main()
