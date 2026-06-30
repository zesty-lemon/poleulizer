import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from dataset_util import annotations_util
from haversine import haversine, Unit
from config import constants as c
from config import paths

STREET_IMAGES_DIR = paths.STREET_IMAGES_DIR

DISTANCE_CUTOFF_M = 140


def normalize_pm180(deg: float) -> float:
    """Wrap any angle in degrees to the [-180, +180] range."""
    return (deg + 180) % 360 - 180


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Return Haversine distance between point 1 and point 2. (haversine = shortest dist between two points on a sphere)
    :param lat1:
    :param lon1:
    :param lat2:
    :param lon2:
    :return:
    """
    point_1 = (lat1, lon1)
    point_2 = (lat2, lon2)

    distance_m = haversine(point_1, point_2, unit=Unit.METERS)
    return distance_m


def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Initial great-circle bearing from point 1 to point 2.

    Works on scalars or array-likes (uses numpy), so the same function backs
    both the per-row FOV gate and vectorized callers.
    :return: degrees clockwise from True North, in the range [0, 360).
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lon_rad = np.radians(np.subtract(lon2, lon1))

    x = np.sin(delta_lon_rad) * np.cos(lat2_rad)
    y = (np.cos(lat1_rad) * np.sin(lat2_rad)
         - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(delta_lon_rad))

    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def relative_bearing_deg(source_lat, source_lon, source_heading,
                         pole_lat, pole_lon, view_yaw=0.0):
    """The pole's signed horizontal angle off the camera's optical axis, degrees.

    + = right of the axis, - = left. This is the single source of truth for
    "theta" — the world bearing source-> pole, minus the vehicle heading and the
    camera's view_yaw, wrapped to [-180, +180]. Works on scalars or array-likes.

    Used by the FOV gate (in_fov), FOV calibration, and pixel projection.

    :param source_heading: vehicle heading, deg CW from True North (EXIF GPSTrack)
    :param view_yaw: camera optical axis vs heading (center=0, left<0, right>0)
    """
    bearing_world = bearing_deg(source_lat, source_lon, pole_lat, pole_lon)
    return normalize_pm180(bearing_world - source_heading - view_yaw)


def predict_x_px(theta_deg, half_fov_deg, image_width_px=c.IMAGE_WIDTH_PX):
    """Pixel column where a pole at angle `theta_deg` off the optical axis should
    appear, under the pinhole (rectilinear) projection.

        x = (W/2) * (1 + tan(theta) / tan(half_fov))

    theta = 0 -> image center (W/2); theta = +half_fov -> right edge (W);
    theta = -half_fov -> left edge (0). Works on scalars or array-likes.

    :param theta_deg:     pole angle off the camera axis (e.g. from relative_bearing_deg)
    :param half_fov_deg:  half the camera's horizontal FOV, degrees (the calibrated value)
    :param image_width_px: image width in pixels (defaults to the dataset constant)
    """
    half_w = image_width_px / 2
    return half_w * (1 + np.tan(np.radians(theta_deg)) / np.tan(np.radians(half_fov_deg)))


def in_fov(
    source_lat: float,
    source_lon: float,
    source_heading: float,
    pole_lat: float,
    pole_lon: float,
    view_yaw: float,
    half_fov: float,
    r_max: float = DISTANCE_CUTOFF_M,
) -> bool:
    """
    Predicate: does the pole at (pole_lat, pole_lon) fall inside the cone of
    a single camera view, given the source pose?

    :param source_lat:     source latitude  (degrees)
    :param source_lon:     source longitude (degrees)
    :param source_heading: source heading,   degrees CW from True North (EXIF GPSTrack)
    :param pole_lat:    labeled pole latitude  (degrees)
    :param pole_lon:    labeled pole longitude (degrees)
    :param view_yaw:    where this camera's optical axis points relative to
                        source heading, deg. center=0, left=negative, right=positive
    :param half_fov:    half the camera's horizontal FOV, degrees
    :param r_max:       drop the pole if the source is farther than this (meters)
    :return: True if the pole is inside (view_yaw +/- half_fov) AND closer than r_max
    """
    # Compute range via haversine_m; early-reject if range > r_max.
    range_m = haversine_m(source_lat, source_lon, pole_lat, pole_lon)
    if range_m > r_max:
        return False

    # Pole's angle off the camera axis (shared helper); inside the cone iff
    # its magnitude is within the half-FOV.
    bearing_camera = relative_bearing_deg(
        source_lat, source_lon, source_heading, pole_lat, pole_lon, view_yaw,
    )
    return abs(bearing_camera) <= half_fov


def retrieve_gps_track(image_name: str) -> float:
    exif_info = read_exif(image_name)
    gps_info = exif_info['GPSInfo']

    # GPSTrack comes back as an IFDRational; cast so callers see a real float.
    source_heading = float(gps_info["GPSTrack"])
    return source_heading


def read_exif(image_name: str) -> dict:
    """
    Open a street_images JPG by filename and return its EXIF tag block.
    :param image_name: filename only (e.g. "000001223389_ROW_20240619.120109.jpg")
    :return: dict of EXIF tag_id -> value (empty dict if the file has no EXIF)
    """
    image_path = STREET_IMAGES_DIR / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    img = Image.open(image_path)

    raw_exif = img._getexif() or {}
    exif: dict = {}
    for tag_id, value in raw_exif.items():
        tag_name = TAGS.get(tag_id, tag_id)
        # GPSInfo is itself a dict of numeric sub-IDs; expand it to readable names.
        if tag_name == "GPSInfo":
            exif["GPSInfo"] = {GPSTAGS.get(k, k): v for k, v in value.items()}
        else:
            exif[tag_name] = value
    return exif


if __name__ == "__main__":
    # Tests

    # Two consecutive source-pose samples from Batch-5 (~8 m apart in real life).
    source_a = (44.41150478488889, -72.01815109669444)
    source_b = (44.411983286574646, -72.01780297878402)
    dist = haversine_m(*source_a, *source_b)
    print(f"haversine_n({source_a}, {source_b}) = {dist:.6f} km  ({dist * 1000:.2f} m)")

    # source_b is roughly north-northeast of source_a -> bearing should be ~20-30 deg.
    bearing = bearing_deg(*source_a, *source_b)
    print(f"bearing_deg({source_a} -> {source_b}) = {bearing:.2f} deg CW from N")

    # in_fov: treat source_a as the source pose, source_b as a candidate pole.
    # source_b sits at bearing ~27.46 deg from source_a, range ~60 m. Sweep the
    # source's heading and which camera we ask about, and check the cone gate.
    pole_lat, pole_lon = source_b
    print("in_fov sanity checks (pole bearing ~27.5 deg from source, range ~60 m):")
    cases = [
        # (label,                     heading, view_yaw, half_fov, r_max, expected)
        ("heading at pole, center cam",  27.46,  0.0,    15.0,    140.0, True),   # pole dead ahead -> in cone
        ("heading north, center cam",     0.0,  0.0,    15.0,    140.0, False),  # pole 27.5 deg right -> outside center cone
        ("heading north, right cam",      0.0, 30.0,    15.0,    140.0, True),   # pole 27.5 deg right -> inside right cone (centered at +30)
        ("heading away from pole",      207.46, 0.0,    15.0,    140.0, False),  # pole behind -> outside
        ("heading at pole, r_max=30 m",  27.46, 0.0,    15.0,     30.0, False),  # range 60 > r_max 30 -> rejected
    ]
    for label, heading, yaw, half, rmax, expected in cases:
        got = in_fov(*source_a, heading, pole_lat, pole_lon, yaw, half, rmax)
        mark = "OK " if got == expected else "FAIL"
        print(f"  [{mark}] {label:35s} -> {got}  (expected {expected})")

    # A known Batch-5 center frame
    # # test we get back the GPSInfo block.
    sample_frame = "000001223389_ROW_20240619.120109.jpg"

    print("Test gps track read")
    print(retrieve_gps_track(sample_frame))

    print("Test image exif read")
    exif = read_exif(sample_frame)
    gps = exif.get("GPSInfo", {})
    print(f"read_exif({sample_frame!r}):")
    print(f"  top-level keys: {list(exif.keys())}")
    print(f"  GPSTrack    = {gps.get('GPSTrack')}")
    print(f"  GPSLatitude = {gps.get('GPSLatitude')} {gps.get('GPSLatitudeRef')}")

