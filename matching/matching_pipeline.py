import pandas as pd

from detection.run_detection import load_detections
from localization.localization_pipeline import load_observations
from scripts.build_pole_catalog import load_pole_catalog


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

    # ----- Match Poles to Bounding Boxes in Each Image -----

    #  cat_row = cat_by_id.loc[some_pole_id]
    for fname in sorted(obs_by_frame):
        # All poles in this frame (from VAIL annotations)
        poles = obs_by_frame[fname]
        # All detected boxes in this frame
        boxes = box_by_frame.get(fname, real.iloc[0:0])  # boxes (empty df if none)

        print(poles)
        print(boxes)


        print("meow")


if __name__ == "__main__":
    perform_matching()

# Todo
"""

1) Cleanup of bounding boxes. Right now they are raw. I need to clean up low confidence / tiny boxes first. (should I do this when I CREATE them instead?)
2) Join these 3 dataframes together somehow? I'd prefer going image by image, and retriving the relevant info from each of the 3 dataframes at each iteration 
"""
