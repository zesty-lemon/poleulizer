from dataset_util import annotations_util
from util import localization_utils


MAX_WALK_BACK = 20
POLE_TYPE = "2"

def localize_poles():
    # Annotations in Route Order
    df = annotations_util.load_annotations()
    print(df.columns)

    for row in df.itertuples():
        # Only stop for annotated poles
        if row.right_rigid_type != POLE_TYPE:
            continue
        # walk backwards from this pole's frame
        for offset in range(1, MAX_WALK_BACK):
            i = row.Index - offset
            if i < 0:
                break

            # row is the pole
            # candidate is the source image
            candidate = df.iloc[i]
            heading_from_exif = localization_utils.retrieve_gps_track(candidate['src_center'])

            is_pole_visible = localization_utils.in_fov(
                source_lat=candidate['source_lat'],
                source_lon=candidate['source_lon'],
                source_heading = heading_from_exif,
                pole_lat=row.right_rigid_lat,
                pole_lon=row.right_rigid_lon,
                view_yaw=0,
                half_fov=15,
                r_max=140
            )

            print(f"Is Pole Visible: {is_pole_visible}")


if __name__ == "__main__":
    localize_poles()


