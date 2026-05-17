import math
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from dataset_util import annotations_util
from haversine import haversine, Unit
from config import constants as c

REPO_ROOT = Path(__file__).resolve().parent.parent
STREET_IMAGES_DIR = REPO_ROOT.parent / "data" / "street_images"

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

    distance_km = haversine(point_1,point_2)

    return distance_km


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Initial great-circle bearing from point 1 to point 2.
    :return: degrees clockwise from True North, in the range [0, 360).
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)

    x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    y = (math.cos(lat1_rad) * math.sin(lat2_rad)
         - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad))

    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


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

    # Two consecutive bus-pose samples from Batch-5 (~8 m apart in real life).
    bus_a = (44.41150478488889, -72.01815109669444)
    bus_b = (44.411983286574646, -72.01780297878402)
    dist = haversine_m(*bus_a, *bus_b)
    print(f"haversine_n({bus_a}, {bus_b}) = {dist:.6f} km  ({dist * 1000:.2f} m)")

    # bus_b is roughly north-northeast of bus_a -> bearing should be ~20-30 deg.
    bearing = bearing_deg(*bus_a, *bus_b)
    print(f"bearing_deg({bus_a} -> {bus_b}) = {bearing:.2f} deg CW from N")

    # A known Batch-5 center frame
    # # test we get back the GPSInfo block.
    sample_frame = "000001223389_ROW_20240619.120109.jpg"
    exif = read_exif(sample_frame)
    gps = exif.get("GPSInfo", {})
    print(f"read_exif({sample_frame!r}):")
    print(f"  top-level keys: {list(exif.keys())}")
    print(f"  GPSTrack    = {gps.get('GPSTrack')}")
    print(f"  GPSLatitude = {gps.get('GPSLatitude')} {gps.get('GPSLatitudeRef')}")
