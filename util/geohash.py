"""Geohash encoder. Inlined to avoid an extra dependency.

Standard geohash base-32 alphabet (omits a, i, l, o to avoid ambiguity).
Precision 9 gives ~5 m cells at Vermont latitudes — the right scale for
distinguishing utility poles at typical on-street spacing.
"""
from __future__ import annotations

_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(lat: float, lon: float, precision: int = 9) -> str:
    """Encode (lat, lon) as a geohash string of the given character length."""
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    bits: list[int] = []
    even = True  # even bits subdivide longitude, odd bits subdivide latitude
    while len(bits) < precision * 5:
        if even:
            mid = (lon_range[0] + lon_range[1]) / 2
            if lon >= mid:
                bits.append(1)
                lon_range[0] = mid
            else:
                bits.append(0)
                lon_range[1] = mid
        else:
            mid = (lat_range[0] + lat_range[1]) / 2
            if lat >= mid:
                bits.append(1)
                lat_range[0] = mid
            else:
                bits.append(0)
                lat_range[1] = mid
        even = not even
    out: list[str] = []
    for i in range(0, len(bits), 5):
        idx = (
            (bits[i] << 4)
            | (bits[i + 1] << 3)
            | (bits[i + 2] << 2)
            | (bits[i + 3] << 1)
            | bits[i + 4]
        )
        out.append(_GEOHASH_BASE32[idx])
    return "".join(out)
