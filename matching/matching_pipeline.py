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
# Cost of leaving a pole unmatched inside the joint assignment (assign_one_to_one).
# Set equal to the gate so the assignment never prefers a pairing the gate would
# reject anyway: a pole is paired only when some box is within this many pixels.
UNMATCHED_COST_PX = GATE_PX
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


def assign_one_to_one(x_pred, x_box, unmatched_cost=None):
    """Joint monotonic alignment of poles -> boxes (minimum total cost).

    Because x_pred is monotonic in theta, the optimal one-to-one matching never
    crosses, so the problem is a sequence alignment: sort poles by x_pred, sort
    boxes by x_box, and find the non-crossing pairing that minimizes

        sum(|x_pred - x_box| over matched pairs) + unmatched_cost * (#unmatched poles)

    Unmatched boxes are free (they are left-of-road / unlabeled poles / false
    positives and are expected). The alignment is solved exactly by dynamic
    programming over the two sorted lists, so every pole's pairing is chosen
    jointly with its neighbours'. The previous greedy left-to-right walk let the
    leftmost pole take its nearest box first and never revisited that choice;
    on the St Johnsbury ground truth that cost ~8 F1 points (one pole stealing
    a neighbour's box accounted for 82 of 270 wrong_box/missed errors).

    unmatched_cost defaults to the module UNMATCHED_COST_PX (= GATE_PX), so a
    pole is only paired when some box lies within the gate; the gate check in
    match_frame then simply confirms it.

    Returns a list of (pole_position, box_position) index pairs into the *input*
    arrays (positional, not DataFrame labels), in pole order.
    """
    if unmatched_cost is None:
        unmatched_cost = UNMATCHED_COST_PX
    n, m = len(x_pred), len(x_box)
    if n == 0 or m == 0:
        return []

    pole_order = sorted(range(n), key=lambda i: x_pred[i])
    box_order = sorted(range(m), key=lambda j: x_box[j])
    xp = [x_pred[i] for i in pole_order]
    xb = [x_box[j] for j in box_order]

    # D[i][j] = min cost of aligning the first i poles with the first j boxes.
    # Moves: skip pole (cost unmatched_cost), skip box (free), match (residual).
    INF = float("inf")
    D = [[INF] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        D[0][j] = 0.0                      # no poles left: remaining boxes are free
    for i in range(1, n + 1):
        D[i][0] = i * unmatched_cost       # no boxes left: every pole unmatched
        for j in range(1, m + 1):
            D[i][j] = min(
                D[i - 1][j - 1] + abs(xp[i - 1] - xb[j - 1]),   # match pole i, box j
                D[i - 1][j] + unmatched_cost,                   # pole i unmatched
                D[i][j - 1],                                    # box j unclaimed
            )

    # Backtrack. On ties prefer a match, then skipping the pole, then the box.
    pairs = []
    i, j = n, m
    while i > 0 and j > 0:
        here = D[i][j]
        if here == D[i - 1][j - 1] + abs(xp[i - 1] - xb[j - 1]):
            pairs.append((pole_order[i - 1], box_order[j - 1]))
            i, j = i - 1, j - 1
        elif here == D[i - 1][j] + unmatched_cost:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def match_frame(fname, poles, boxes):
    """Predict, score, assign, and gate one frame. Returns a list of match rows
    (one per expected pole, including unmatched ones)."""
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
            if residual <= GATE_PX:
                rows.append(row(pole, "matched", box, residual))
                matched_poles.add(pi)

    for pi, pole in enumerate(pole_list):
        if pi not in matched_poles:
            rows.append(row(pole, "unmatched_no_box"))

    return rows


def perform_matching():
    # ----- Load Data -----
    # Pole Observations - Which frames show which pole
    observations_df = load_observations()
    # Pole Detections - Bounding Boxes per image
    detections_df = load_detections()

    # ----- Build Indexes -----
    obs_by_frame = dict(tuple(observations_df.groupby("image_filename")))  # frame -> its poles
    # Drop zero-detection sentinels and low-confidence boxes.
    real = detections_df[(detections_df.box_index >= 0) & (detections_df.score >= MIN_SCORE)]
    box_by_frame = dict(tuple(real.groupby("image_filename")))  # frame -> its boxes

    # ----- Match Poles to Bounding Boxes in Each Image -----
    all_rows = []
    for fname in sorted(obs_by_frame):
        poles = obs_by_frame[fname]
        boxes = box_by_frame.get(fname, real.iloc[0:0])  # boxes (empty df if none)
        all_rows.extend(match_frame(fname, poles, boxes))

    matches = pd.DataFrame(all_rows)

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
