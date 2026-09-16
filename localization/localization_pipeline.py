import pandas as pd

from config import paths
from dataset_util import annotations_util
from util import localization_utils
from util.geohash import encode_geohash

MAX_WALK_BACK = 20
POLE_TYPE = "2"

# Center camera placeholder geometry; measured intrinsics to come.
CENTER_YAW_DEG = 0.0

CENTER_HALF_FOV_DEG = 19.5

OBS_OUT_PATH = paths.OBSERVATIONS_PATH


def localize_poles() -> pd.DataFrame:
    """Build the (pole_id, frame_id) observation table for every right-labeled pole,
    using the center camera only."""
    df = annotations_util.load_annotations().reset_index(drop=True)
    # 1-indexed position of each frame within its batch, matching the annotation tool's
    # frame counter. Write a frame_number into the output for visual lookup.
    df["frame_number"] = df.groupby("batch").cumcount() + 1
    print(f"loaded {len(df)} annotations across {df.batch.nunique()} batch(es)")

    heading_cache: dict[str, float] = {}
    pole_index_by_id: dict[str, int] = {}
    rows: list[dict] = []

    for label_row in df.itertuples():
        if label_row.right_rigid_type != POLE_TYPE:
            continue
        pole_lat = label_row.right_rigid_lat
        pole_lon = label_row.right_rigid_lon
        if pd.isna(pole_lat) or pd.isna(pole_lon):
            continue
        pole_id = "geohash:" + encode_geohash(pole_lat, pole_lon)
        # Temporary 1, 2, 3... counter assigned in order of first appearance.
        # Easier to manually track while clicking through observations than the geohash.
        if pole_id not in pole_index_by_id:
            pole_index_by_id[pole_id] = len(pole_index_by_id) + 1
        pole_index = pole_index_by_id[pole_id]

        # Walk back from this pole's label frame; record any frame the pole appears in.
        for offset in range(1, MAX_WALK_BACK + 1):
            prior_pos = label_row.Index - offset
            if prior_pos < 0:
                break
            candidate = df.iloc[prior_pos]
            if candidate["batch"] != label_row.batch:
                break

            center_fname = candidate["src_center"]
            if center_fname not in heading_cache:
                heading_cache[center_fname] = localization_utils.retrieve_gps_track(center_fname)
            heading = heading_cache[center_fname]

            if not localization_utils.in_fov(
                source_lat=candidate["source_lat"],
                source_lon=candidate["source_lon"],
                source_heading=heading,
                pole_lat=pole_lat,
                pole_lon=pole_lon,
                view_yaw=CENTER_YAW_DEG,
                half_fov=CENTER_HALF_FOV_DEG,
            ):
                continue
            rows.append({
                "pole_index":     pole_index,
                "frame_number":   int(candidate["frame_number"]),
                "pole_id":        pole_id,
                "pole_lat":       pole_lat,
                "pole_lon":       pole_lon,
                "frame_id":       candidate["filename"],
                "image_filename": candidate["src_center"],
                "source_lat":     candidate["source_lat"],
                "source_lon":     candidate["source_lon"],
                "source_heading": heading,
                "batch":          candidate["batch"],
            })

    obs = pd.DataFrame(rows)
    # Same physical pole annotated in multiple frames -> same geohash;
    # collapse repeated (pole_id, frame_id) pairs.
    obs = obs.drop_duplicates(subset=["pole_id", "frame_id"])
    # Group all observations of pole 1 together, ordered by their frame number, then pole 2, etc.
    obs = obs.sort_values(["pole_index", "frame_number"]).reset_index(drop=True)
    return obs


def load_observations() -> pd.DataFrame:
    """Load the localization observations table (one row per (pole_id, frame_id)).

    Raises FileNotFoundError if localization hasn't been run yet — run
    localize_poles (the `localize_poles` pipeline step) first.
    """
    if not OBS_OUT_PATH.exists():
        raise FileNotFoundError(
            f"No observations at {OBS_OUT_PATH}; run localize_poles first"
        )
    return pd.read_parquet(OBS_OUT_PATH)


def run_localization_pipeline():
    obs = localize_poles()
    OBS_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    obs.to_parquet(OBS_OUT_PATH, compression="gzip", index=False)
    csv_path = OBS_OUT_PATH.with_suffix(".csv")
    obs.to_csv(csv_path, index=False)

    print(f"\nwrote {len(obs)} observations to:")
    print(f"  {OBS_OUT_PATH}")
    print(f"  {csv_path}")
    print(f"  unique poles:  {obs.pole_id.nunique()}")
    print(f"  unique frames: {obs.frame_id.nunique()}")

def main():
    run_localization_pipeline()

if __name__ == "__main__":
    main()
