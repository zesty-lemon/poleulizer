import pandas as pd

from config import paths
from detection.run_detection import load_detections
from localization.localization_pipeline import load_observations, CENTER_HALF_FOV_DEG
from scripts.build_pole_catalog import load_pole_catalog
from util import localization_utils as lu
from util.localization_utils import predict_x_px

# Matching policy. GATE_PX: a pole's matched box must land within this many pixels
# of its prediction. Originally ~2x the FOV-calibration residual (~78 px) = 150;
# raised to 225 by the F1-optimal point of the offline sweep against the
# St Johnsbury ground truth (evaluate.py sweep; recall 47.5%->54.0% for ~1pt
# precision). THETA_MAX drops poles near the FOV edge, where the rectilinear model
# / correspondence break down (see documentation/matching_design.md).
GATE_PX = 225.0
THETA_MAX = 18.0
# Drop detector boxes below this confidence before matching (kept raw in
# bboxes.parquet; filtered here because it's a matching policy, not a data fact).
MIN_SCORE = 0.3


def append_bearings_onto_df_row(pole_row):
    """
    Calculate theta (bearing) from each pole to its source lat/long
    :param pole_row:
    :return:
    """
    theta = lu.relative_bearing_deg(
        pole_row.source_lat,
        pole_row.source_lon,
        pole_row.source_heading,
        pole_row.pole_lat,
        pole_row.pole_lon,
        view_yaw=0.0
    )
    return theta


def assign_one_to_one(x_pred, x_box):
    """Monotonic sorted-alignment of poles -> boxes.

    Because x_pred is monotonic in theta, the optimal matching never crosses, so
    we can sort both sides and walk them together, taking the nearest available
    box for each pole. Returns a list of (pole_position, box_position) index
    pairs into the *input* arrays (positional, not DataFrame labels).
    """
    pole_order = sorted(range(len(x_pred)), key=lambda i: x_pred[i])
    box_order = sorted(range(len(x_box)), key=lambda j: x_box[j])

    pairs = []
    b = 0
    for pi in pole_order:
        # Advance through boxes; greedily take the closest still-available box.
        best_bj, best_d = None, None
        for k in range(b, len(box_order)):
            bj = box_order[k]
            d = abs(x_pred[pi] - x_box[bj])
            if best_d is None or d < best_d:
                best_bj, best_d, best_k = bj, d, k
            else:
                # Distances are V-shaped in sorted order; once they grow, stop.
                break
        if best_bj is not None:
            pairs.append((pi, best_bj))
            b = best_k + 1  # consume boxes up to and including the chosen one
    return pairs


def match_frame(fname, poles, boxes, gate_px=None):
    """Predict, score, assign, and gate one frame. Returns a list of match rows
    (one per expected pole, including unmatched ones).

    gate_px overrides the module GATE_PX for this call (used by the offline
    parameter sweep, which runs with gate_px=inf to get every assigned pole's
    residual before any gating). Defaults to the module GATE_PX."""
    gate = GATE_PX if gate_px is None else gate_px
    # Predict each pole's pixel column from its angle off the optical axis.
    poles = poles.assign(theta=poles.apply(append_bearings_onto_df_row, axis=1))
    poles = poles.assign(x_pred=predict_x_px(poles.theta, CENTER_HALF_FOV_DEG))

    boxes = boxes.assign(x_box=(boxes.bbox_x1 + boxes.bbox_x2) / 2)

    # Edge skip: poles beyond the trustworthy cone aren't matched.
    in_cone = poles.theta.abs() <= THETA_MAX
    edge_poles = poles[~in_cone]
    cone_poles = poles[in_cone]

    rows = []

    def row(pole, status, box=None, residual=None):
        return {
            # Frame IDs: human-readable counter + the actual filename.
            "frame_number": int(pole.frame_number),
            "image_filename": fname,
            # Pole IDs: human-readable counter + the actual geohash id.
            "pole_index": pole.pole_index,
            "pole_id": pole.pole_id,
            "box_index": (int(box.box_index) if box is not None else None),
            "theta_deg": float(pole.theta),
            "x_pred": float(pole.x_pred),
            "x_box": (float(box.x_box) if box is not None else None),
            "residual_px": (float(residual) if residual is not None else None),
            "status": status,
            "batch": pole.batch,
        }

    for pole in edge_poles.itertuples():
        rows.append(row(pole, "unmatched_edge"))

    # Assign cone poles to boxes one-to-one, then gate on pixel distance.
    pole_list = list(cone_poles.itertuples())
    box_list = list(boxes.itertuples())
    matched_poles = set()
    if pole_list and box_list:
        x_pred = [p.x_pred for p in pole_list]
        x_box = [b.x_box for b in box_list]
        for pi, bj in assign_one_to_one(x_pred, x_box):
            pole, box = pole_list[pi], box_list[bj]
            residual = abs(pole.x_pred - box.x_box)
            if residual <= gate:
                rows.append(row(pole, "matched", box, residual))
                matched_poles.add(pi)

    for pi, pole in enumerate(pole_list):
        if pi not in matched_poles:
            rows.append(row(pole, "unmatched_no_box"))

    return rows


def match_all(gate_px=None, min_score=MIN_SCORE):
    """Run the matcher over every frame and return the match table (no writing).

    Shared by perform_matching (default params) and the offline sweep (which
    passes gate_px=inf to recover every assigned pole's residual pre-gating)."""
    # ----- Load Data -----
    # Pole Observations - Which frames show which pole
    observations_df = load_observations()
    # Pole Detections - Bounding Boxes per image
    detections_df = load_detections()

    # ----- Build Indexes -----
    obs_by_frame = dict(tuple(observations_df.groupby("image_filename")))  # frame -> its poles
    # Drop zero-detection sentinels and low-confidence boxes.
    real = detections_df[(detections_df.box_index >= 0) & (detections_df.score >= min_score)]
    box_by_frame = dict(tuple(real.groupby("image_filename")))  # frame -> its boxes

    # ----- Match Poles to Bounding Boxes in Each Image -----
    all_rows = []
    for fname in sorted(obs_by_frame):
        poles = obs_by_frame[fname]
        boxes = box_by_frame.get(fname, real.iloc[0:0])  # boxes (empty df if none)
        all_rows.extend(match_frame(fname, poles, boxes, gate_px=gate_px))

    return pd.DataFrame(all_rows)


def perform_matching():
    matches = match_all()

    # ----- Write -----
    paths.MATCHES_PATH.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(paths.MATCHES_PATH, compression="gzip", index=False)

    # Human-readable CSV mirror (rounded floats) for manual review / comparison
    # against hand annotations.
    csv_path = paths.MATCHES_PATH.with_suffix(".csv")
    matches.round({"theta_deg": 1, "x_pred": 0, "x_box": 0, "residual_px": 1}) \
           .to_csv(csv_path, index=False)

    n_matched = (matches.status == "matched").sum()
    n_poles = len(matches)
    print(f"wrote {n_poles} pole-rows to:")
    print(f"  {paths.MATCHES_PATH}")
    print(f"  {csv_path}")
    print(f"  matched:          {n_matched} ({n_matched / n_poles:.0%})")
    print(f"  unmatched_no_box: {(matches.status == 'unmatched_no_box').sum()}")
    print(f"  unmatched_edge:   {(matches.status == 'unmatched_edge').sum()}")
    if n_matched:
        print(f"  median residual:  {matches.loc[matches.status == 'matched', 'residual_px'].median():.1f} px")
    return matches


def load_matches() -> pd.DataFrame:
    """Load the pole<->box match table. Raises if matching hasn't been run."""
    if not paths.MATCHES_PATH.exists():
        raise FileNotFoundError(
            f"No matches at {paths.MATCHES_PATH}; run perform_matching first"
        )
    return pd.read_parquet(paths.MATCHES_PATH)


if __name__ == "__main__":
    perform_matching()
