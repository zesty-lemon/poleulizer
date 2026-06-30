import pandas as pd

from detection.run_detection import load_detections
from localization.localization_pipeline import load_observations, CENTER_HALF_FOV_DEG
from scripts.build_pole_catalog import load_pole_catalog
from util import localization_utils as lu
from util.localization_utils import predict_x_px


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


def perform_matching():
    # ----- Load Data -----
    # Pole Observations - Which frames show which pole
    observations_df = load_observations()
    # Pole Detections - Bounding Boxes per image
    detections_df = load_detections()
    # Pole Catalog - Master list of poles
    catalog_df = load_pole_catalog()

    # ----- Build Indexes -----
    obs_by_frame = dict(tuple(observations_df.groupby("image_filename")))  # frame -> its poles
    real = detections_df[detections_df.box_index >= 0]
    box_by_frame = dict(tuple(real.groupby("image_filename")))  # frame -> its boxes
    cat_by_id = catalog_df.set_index("pole_id")  # pole_id -> catalog row (lookup)
    #  cat_row = cat_by_id.loc[some_pole_id]

    # ----- Match Poles to Bounding Boxes in Each Image -----
    for fname in sorted(obs_by_frame):
        # All poles in this frame (from VAIL annotations)
        poles = obs_by_frame[fname]
        # All detected boxes in this frame
        boxes = box_by_frame.get(fname, real.iloc[0:0])  # boxes (empty df if none)

        if boxes.empty:
            continue

        # Get and append theta to each pole from the current frame
        poles = poles.assign(theta=poles.apply(append_bearings_onto_df_row, axis=1))

        # Get and append predicted x pixel values to each pole
        poles = poles.assign(x_pred=predict_x_px(poles.theta, CENTER_HALF_FOV_DEG))

        


if __name__ == "__main__":
    perform_matching()

# Todo
"""

1) Cleanup of bounding boxes. Right now they are raw. I need to clean up low confidence / tiny boxes first. (should I do this when I CREATE them instead?)
2) Join these 3 dataframes together somehow? I'd prefer going image by image, and retriving the relevant info from each of the 3 dataframes at each iteration 
"""
