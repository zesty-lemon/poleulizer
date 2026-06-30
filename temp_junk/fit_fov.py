"""Calibrate the center camera's horizontal FOV from existing pipeline data.

Estimates HFOV (and a lens-distortion flag) by reconciling two independent
locators of each pole's horizontal position:

    geometry (GPS)  ->  real-world angle theta off the optical axis
    detector        ->  pixel column x  ->  x_norm in [-1, +1]

The pinhole relation `x_norm = tan(theta) / tan(HFOV/2)` is linear in
`k = 1/tan(HFOV/2)`, so the calibration is a (robust, weighted) slope fit.

No images, no model, no manual annotation. Reads two parquets, prints a report.
See documentation/fov_calibration_design.md for the full design, including the
side-asymmetry handling and the gate-coupling caveat.

Run:
    python3 scripts/fit_fov.py --yaw-deg 0 --out fov.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Put the repo root on the path so local packages are importable as a script.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths  # noqa: E402
from util import localization_utils as lu  # noqa: E402

# Image geometry. Confirmed uniform from EXIF (3840x2160); EXIF carries no
# focal/FOV/lens tags, only GPS — hence this empirical calibration.
W = 3840
CX = W / 2

HAND_MEASURED_HFOV_DEG = 45.8  # one-pole hand measurement, for cross-check
EARTH_RADIUS_M = 6_371_000.0


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def relative_bearing_deg(obs: pd.DataFrame) -> pd.Series:
    """theta per observation row: the pole's signed horizontal angle off the
    vehicle heading, in degrees. + = right of axis. Thin DataFrame adapter over
    the shared, array-safe localization_utils.relative_bearing_deg."""
    theta = lu.relative_bearing_deg(
        obs["source_lat"].to_numpy(),
        obs["source_lon"].to_numpy(),
        obs["source_heading"].to_numpy(),
        obs["pole_lat"].to_numpy(),
        obs["pole_lon"].to_numpy(),
    )
    return pd.Series(theta, index=obs.index)


def range_m(obs: pd.DataFrame) -> pd.Series:
    """Great-circle distance source->pole in meters (vectorized haversine)."""
    slat = np.radians(obs["source_lat"].to_numpy())
    plat = np.radians(obs["pole_lat"].to_numpy())
    dlat = plat - slat
    dlon = np.radians((obs["pole_lon"] - obs["source_lon"]).to_numpy())
    a = np.sin(dlat / 2) ** 2 + np.cos(slat) * np.cos(plat) * np.sin(dlon / 2) ** 2
    d = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return pd.Series(d, index=obs.index)


def box_centers(boxes: pd.DataFrame, min_score: float) -> pd.DataFrame:
    """Drop sentinel (box_index == -1) and low-score boxes; add x_center, x_norm."""
    real = boxes[boxes["box_index"] >= 0].copy()
    if min_score > 0:
        real = real[real["score"].fillna(0.0) >= min_score]
    real["x_center"] = (real["bbox_x1"] + real["bbox_x2"]) / 2.0
    real["x_norm"] = (real["x_center"] - CX) / CX
    return real


# --------------------------------------------------------------------------- #
# Side asymmetry & frame selection (prerequisite — see design doc)
# --------------------------------------------------------------------------- #
def _straight_frames(obs: pd.DataFrame, max_curvature_deg: float) -> set[str]:
    """Image filenames whose local heading-change rate (deg per frame step,
    measured against batch neighbors) is at or below the threshold."""
    frames = (
        obs[["image_filename", "batch", "frame_number", "source_heading"]]
        .drop_duplicates("image_filename")
        .sort_values(["batch", "frame_number"])
        .reset_index(drop=True)
    )
    g = frames.groupby("batch", sort=False)
    prev_h = g["source_heading"].shift()
    prev_fn = g["frame_number"].shift()
    dh = ((frames["source_heading"] - prev_h + 180.0) % 360.0) - 180.0
    dfn = (frames["frame_number"] - prev_fn).abs().clip(lower=1)
    frames["rate_prev"] = (dh.abs() / dfn)
    # Rate to the next frame is the following row's rate_prev, within the batch.
    frames["rate_next"] = frames.groupby("batch", sort=False)["rate_prev"].shift(-1)
    # Edge frames have one NaN neighbor; judge them on the side that exists.
    curvature = frames[["rate_prev", "rate_next"]].max(axis=1, skipna=True)
    keep = frames["image_filename"][curvature.fillna(0.0) <= max_curvature_deg]
    return set(keep)


def select_frames(
    obs: pd.DataFrame,
    boxes: pd.DataFrame,
    *,
    max_curvature_deg: float,
    center_band: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Side-asymmetry prerequisite. Filters on independent geometry only (never
    on the fit residual), so discarding data costs leverage but not bias:

      1. straight-segment gate  — drop curved frames (where "right road" no
         longer implies "right image"),
      2. theta-sign side match  — annotated poles are right-of-road, so on
         straight frames they sit at theta > 0; keep only right-side boxes,
      3. center-band exclusion  — drop the near-vanishing-point overlap zone.
    """
    straight = _straight_frames(obs, max_curvature_deg)

    obs_f = obs[obs["image_filename"].isin(straight) & (obs["theta_deg"] > 0)].copy()
    boxes_f = boxes[
        boxes["image_filename"].isin(straight) & (boxes["x_norm"] > center_band)
    ].copy()
    return obs_f, boxes_f


# --------------------------------------------------------------------------- #
# Correspondence
# --------------------------------------------------------------------------- #
def build_pairs(
    obs: pd.DataFrame,
    boxes: pd.DataFrame,
    *,
    mode: str,
    max_frame_poles: int | None,
) -> pd.DataFrame:
    """One row per matched (pole, box). Assumes select_frames() already applied,
    so theta > 0 and x_norm > 0 for every survivor (sign match is enforced again
    here as a cheap guard)."""
    pole_cols = ["image_filename", "pole_id", "theta_deg", "range_m"]
    poles = obs[pole_cols].drop_duplicates(["image_filename", "pole_id"])

    rows: list[dict] = []
    box_by_frame = dict(tuple(boxes.groupby("image_filename")))
    for fname, pole_grp in poles.groupby("image_filename"):
        box_grp = box_by_frame.get(fname)
        if box_grp is None or box_grp.empty:
            continue
        n_poles, n_boxes = len(pole_grp), len(box_grp)
        if max_frame_poles is not None and n_poles > max_frame_poles:
            continue

        p_sorted = pole_grp.sort_values("theta_deg")
        b_sorted = box_grp.sort_values("x_norm")

        if mode == "rank":
            if n_poles != n_boxes:
                continue
            paired = zip(p_sorted.itertuples(), b_sorted.itertuples())
        elif mode == "extreme":
            # Right (max) extreme only — the leftmost box is typically an
            # unannotated left-of-road distractor.
            paired = [(p_sorted.iloc[-1:].itertuples().__next__(),
                       b_sorted.iloc[-1:].itertuples().__next__())]
        else:
            raise ValueError(f"unknown mode {mode!r}")

        for p, b in paired:
            # Defensive sign guard (no-op after select_frames, matters if reused).
            if math.copysign(1, math.tan(math.radians(p.theta_deg))) != math.copysign(1, b.x_norm):
                continue
            rows.append({
                "image_filename": fname,
                "pole_id": p.pole_id,
                "box_index": int(b.box_index),
                "theta_deg": float(p.theta_deg),
                "x_norm": float(b.x_norm),
                "range_m": float(p.range_m),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def _wls(t: np.ndarray, x: np.ndarray, w: np.ndarray, fit_yaw: bool) -> tuple[float, float]:
    """Weighted least squares. Returns (k, c). c is 0 unless fit_yaw."""
    if fit_yaw:
        # Solve [Σwt² Σwt; Σwt Σw] [k; c] = [Σwtx; Σwx]
        Stt = np.sum(w * t * t)
        St = np.sum(w * t)
        Sw = np.sum(w)
        Stx = np.sum(w * t * x)
        Sx = np.sum(w * x)
        A = np.array([[Stt, St], [St, Sw]])
        rhs = np.array([Stx, Sx])
        k, c = np.linalg.solve(A, rhs)
        return float(k), float(c)
    k = np.sum(w * t * x) / np.sum(w * t * t)
    return float(k), 0.0


def fit_fov(
    pairs: pd.DataFrame,
    *,
    yaw_deg: float,
    fit_yaw: bool,
    weight: str,
    robust_sigma: float,
    max_angle_deg: float,
    distortion_tol: float,
    min_pairs: int = 20,
) -> dict:
    """Weighted robust slope fit -> result dict (schema in the design doc)."""
    df = pairs[pairs["theta_deg"].abs() <= max_angle_deg].copy()
    if df.empty:
        raise ValueError("no pairs survive the angular guard")

    # Independent variable: tan(theta), with fixed yaw subtracted unless co-fitting.
    theta_rad = np.radians(df["theta_deg"].to_numpy() - (0.0 if fit_yaw else yaw_deg))
    t = np.tan(theta_rad)
    x = df["x_norm"].to_numpy()
    w = (df["range_m"].to_numpy() ** 2) if weight == "range2" else np.ones(len(df))

    n_built = len(df)
    mask = np.ones(len(df), dtype=bool)
    k = c = 0.0
    for _ in range(5):
        k, c = _wls(t[mask], x[mask], w[mask], fit_yaw)
        resid = x - (k * t + c)
        sigma = math.sqrt(np.sum(w[mask] * resid[mask] ** 2) / np.sum(w[mask]))
        new_mask = np.abs(resid) <= robust_sigma * sigma
        if new_mask.sum() == mask.sum():
            mask = new_mask
            break
        mask = new_mask
        if mask.sum() < 2:
            raise ValueError("robust rejection removed too many pairs")

    if k <= 0:
        raise ValueError(f"fitted slope k={k:.4f} <= 0 — sign convention or data bug")

    resid = x - (k * t + c)
    used_resid = resid[mask]
    hfov_deg = 2 * math.degrees(math.atan(1.0 / k))

    # Distortion check: does the residual trend with position?
    xm = x[mask]
    M = np.column_stack([xm, xm ** 3])
    cubic_coef = float(np.linalg.lstsq(M, used_resid, rcond=None)[0][1]) if mask.sum() > 3 else 0.0
    corr = float(np.corrcoef(np.abs(xm), used_resid)[0, 1]) if mask.sum() > 3 else 0.0
    distortion_flag = (abs(cubic_coef) > distortion_tol) or (abs(corr) > 0.3)

    result = {
        "hfov_deg": round(hfov_deg, 3),
        "half_fov_deg": round(hfov_deg / 2, 3),
        "fx_px": round(CX * k, 1),
        "yaw_deg_applied": (None if fit_yaw else yaw_deg),
        "yaw_offset_deg": (round(-math.degrees(math.atan(c / k)), 3) if fit_yaw else None),
        "slope_k": round(k, 6),
        "intercept_c": round(c, 6),
        "n_pairs_built": int(n_built),
        "n_used": int(mask.sum()),
        "n_dropped": int(n_built - mask.sum()),
        "residual_px_std": round(float(np.std(used_resid)) * CX, 1),
        "distortion_flag": bool(distortion_flag),
        "distortion_cubic_coef": round(cubic_coef, 4),
        "mode": None,        # filled by main()
        "weight": weight,
        "robust_sigma": robust_sigma,
        "hand_measured_hfov_deg": HAND_MEASURED_HFOV_DEG,
        "_low_confidence": bool(mask.sum() < min_pairs),
    }
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _gate_coupling_warning(pairs: pd.DataFrame) -> str | None:
    """Heuristic: if many pairs pile up at the max observed |theta|, the input
    observations were likely truncated by the in_fov gate — regenerate wider."""
    if pairs.empty:
        return None
    abs_theta = pairs["theta_deg"].abs()
    ceiling = abs_theta.max()
    frac_at_ceiling = (abs_theta >= ceiling - 1.0).mean()
    if frac_at_ceiling > 0.10:
        return (
            f"{frac_at_ceiling:.0%} of pairs pile up within 1° of max |θ|={ceiling:.1f}° — "
            "observations may be gate-truncated. Regenerate with a wider "
            "CENTER_HALF_FOV_DEG (e.g. 35) before trusting this fit (see design doc)."
        )
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate center-camera horizontal FOV.")
    ap.add_argument("--mode", choices=["rank", "extreme"], default="rank")
    ap.add_argument("--max-curvature-deg", type=float, default=3.0)
    ap.add_argument("--center-band", type=float, default=0.05)
    ap.add_argument("--yaw-deg", type=float, default=0.0)
    ap.add_argument("--fit-yaw", action="store_true",
                    help="co-fit intercept instead of fixing yaw (unreliable: one-sided data)")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--max-frame-poles", type=int, default=None)
    ap.add_argument("--max-angle-deg", type=float, default=40.0)
    ap.add_argument("--weight", choices=["range2", "none"], default="range2")
    ap.add_argument("--robust-sigma", type=float, default=3.0)
    ap.add_argument("--distortion-tol", type=float, default=0.02)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--scatter", type=Path, default=None)
    args = ap.parse_args()

    obs = pd.read_parquet(paths.OBSERVATIONS_PATH)
    boxes_raw = pd.read_parquet(paths.BBOX_PATH)

    # Drop rows missing the coords we need, then attach geometry.
    n0 = len(obs)
    obs = obs.dropna(subset=["pole_lat", "pole_lon", "source_lat", "source_lon", "source_heading"])
    if len(obs) < n0:
        print(f"dropped {n0 - len(obs)} observation rows with NaN coords")
    obs = obs.assign(theta_deg=relative_bearing_deg(obs), range_m=range_m(obs))

    boxes = box_centers(boxes_raw, args.min_score)

    obs_f, boxes_f = select_frames(
        obs, boxes,
        max_curvature_deg=args.max_curvature_deg,
        center_band=args.center_band,
    )
    print(f"frames after straight/side/band filter: "
          f"{obs_f['image_filename'].nunique()} kept "
          f"(of {obs['image_filename'].nunique()}); "
          f"poles={len(obs_f)} boxes={len(boxes_f)}")

    pairs = build_pairs(obs_f, boxes_f, mode=args.mode, max_frame_poles=args.max_frame_poles)
    if pairs.empty:
        print("error: no pairs built.", file=sys.stderr)
        if args.mode == "rank":
            print("  no count-matched frames — try --mode extreme", file=sys.stderr)
        return 1
    print(f"built {len(pairs)} ({args.mode}) candidate pairs")

    try:
        result = fit_fov(
            pairs,
            yaw_deg=args.yaw_deg,
            fit_yaw=args.fit_yaw,
            weight=args.weight,
            robust_sigma=args.robust_sigma,
            max_angle_deg=args.max_angle_deg,
            distortion_tol=args.distortion_tol,
        )
    except ValueError as e:
        print(f"error: fit failed — {e}", file=sys.stderr)
        return 1
    result["mode"] = args.mode

    # --- report ---
    delta = result["hfov_deg"] - HAND_MEASURED_HFOV_DEG
    print("\n----- FOV calibration -----")
    print(f"  HFOV         = {result['hfov_deg']}°   (half = {result['half_fov_deg']}°)")
    print(f"  fx           = {result['fx_px']} px")
    if args.fit_yaw:
        print(f"  yaw offset   = {result['yaw_offset_deg']}°  "
              f"** WARNING: one-sided θ coverage — treat as unreliable **")
    else:
        print(f"  yaw applied  = {result['yaw_deg_applied']}° (fixed)")
    print(f"  pairs used   = {result['n_used']} (dropped {result['n_dropped']} as outliers)")
    print(f"  residual     = {result['residual_px_std']} px (matcher gating budget)")
    print(f"  distortion   = {'YES' if result['distortion_flag'] else 'no'} "
          f"(cubic coef {result['distortion_cubic_coef']})")
    print(f"\n  fitted HFOV = {result['hfov_deg']}°  "
          f"(hand-measured {HAND_MEASURED_HFOV_DEG}°,  Δ = {delta:+.1f}°)")

    warn = _gate_coupling_warning(pairs)
    if warn:
        print(f"\n  ⚠️  {warn}")
    if result["_low_confidence"]:
        print(f"\n  ⚠️  only {result['n_used']} pairs used — low-confidence estimate")

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")
    if args.scatter:
        keep = pairs[pairs["theta_deg"].abs() <= args.max_angle_deg].copy()
        keep["tan_theta"] = np.tan(np.radians(keep["theta_deg"] - (0.0 if args.fit_yaw else args.yaw_deg)))
        keep.to_csv(args.scatter, index=False)
        print(f"wrote {args.scatter}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
