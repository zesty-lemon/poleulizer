from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from dataset_util import annotations_util
from haversine import haversine, Unit
from config import constants as c

REPO_ROOT = Path(__file__).resolve().parent.parent
STREET_IMAGES_DIR = REPO_ROOT.parent / "data" / "street_images"

def haversine_m(lat1, lon1, lat2, lon2) -> float:
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


def localize_poles():
    # Load Pole Annotations
    annotations_df = annotations_util.load_annotations()
    print("meow")

if __name__ == "__main__":
    localize_poles()

    # Two consecutive bus-pose samples from Batch-5 (~8 m apart in real life).
    bus_a = (44.41150, -72.01815)
    bus_b = (44.41157, -72.01811)
    dist = haversine_m(*bus_a, *bus_b)
    print(f"haversine_n({bus_a}, {bus_b}) = {dist:.6f} km  ({dist * 1000:.2f} m)")
