from dataset_util import annotations_util
from haversine import haversine, Unit
from config import constants as c

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

def localize_poles():
    # Load Pole Annotations
    annotations_df = annotations_util.load_annotations()


if __name__ == "__main__":
    localize_poles()

    # Two consecutive bus-pose samples from Batch-5 (~8 m apart in real life).
    bus_a = (44.41150, -72.01815)
    bus_b = (44.41157, -72.01811)
    dist = haversine_m(*bus_a, *bus_b)
    print(f"haversine_n({bus_a}, {bus_b}) = {dist:.6f} km  ({dist * 1000:.2f} m)")
