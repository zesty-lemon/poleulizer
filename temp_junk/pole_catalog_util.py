"""Query interface over the pole catalog parquet.

Wraps the on-disk catalog (built by `scripts/build_pole_catalog.py`) with the
lookup patterns documented in `documentation/pole_catalog.md` and matching the
Phase 3 contract from `pole_identity_design.md`.

Lazy-loads the parquet on first query and caches the DataFrame in memory — for
batch matching, construct one PoleCatalog and reuse it.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


# Paths resolve from this file's location so the module works from any CWD.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT.parent / "data"
DEFAULT_CATALOG_PATH = DATA_DIR / "pole_catalog" / "poles.parquet"

# Per pole_identity_design.md: sized for 1-5m utility GPS error stacked with
# projection error from the Phase 1 ray back-projection.
DEFAULT_MATCH_GATE_M = 10.0

# Geohash-9 cells are ~5m at VT latitudes — coarse enough to absorb sub-cell
# GPS jitter on repeat observations of the same pole.
DEFAULT_GEOHASH_PRECISION = 9

# Standard geohash base-32 alphabet (omits a, i, l, o to avoid ambiguity).
_GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def _encode_geohash(lat: float, lon: float, precision: int = 9) -> str:
    """Encode (lat, lon) as a geohash string. Inlined to avoid an extra dep."""
    lat_range = [-90.0, 90.0]
    lon_range = [-180.0, 180.0]
    bits: list[int] = []
    even = True  # alternating: even bits subdivide longitude, odd subdivide latitude
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


class PoleCatalog:
    """Lookup interface over the pole catalog parquet.

    Typical use::

        cat = PoleCatalog()
        cat.find_by_pole_id("GMP:4521A37")              # exact key lookup
        cat.find_by_natural_id("4521A37")               # by utility's own ID
        cat.nearest(44.4115, -72.0182)                  # spatial: 1 best match
        cat.candidates_within(44.4115, -72.0182, k=5)   # spatial: top-k for ambiguity
        cat.match_or_provisional(44.4115, -72.0182)     # Phase 3 contract
        cat.in_bbox(44.40, 44.45, -72.05, -72.00)       # region filter
    """

    def __init__(self, parquet_path: Path | str | None = None):
        self.parquet_path = (
            Path(parquet_path) if parquet_path is not None else DEFAULT_CATALOG_PATH
        )
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        """Lazy-loaded DataFrame view of the catalog. First access pays the I/O cost."""
        if self._df is None:
            self._df = pd.read_parquet(self.parquet_path)
        return self._df

    # ----- exact-match lookups -----

    def find_by_pole_id(self, pole_id: str) -> dict[str, Any] | None:
        """Full row for a pole_id, or None if not in the catalog."""
        hits = self.df[self.df.pole_id == pole_id]
        return hits.iloc[0].to_dict() if not hits.empty else None

    def find_by_natural_id(
        self, natural_id: str, utility: str | None = None
    ) -> str | None:
        """Canonical pole_id given a utility's own ID (POLETAG or gs_guid).

        Provide ``utility`` to disambiguate if the same natural_id appears in
        multiple layers (rare but theoretically possible).
        """
        rows = self.df[self.df.natural_id == natural_id]
        if utility is not None:
            rows = rows[rows.utility == utility]
        return rows.pole_id.iloc[0] if not rows.empty else None

    # ----- spatial lookups -----

    def in_bbox(
        self, lat_min: float, lat_max: float, lon_min: float, lon_max: float
    ) -> pd.DataFrame:
        """All poles inside a WGS84 bounding box. Useful as a pre-filter."""
        return self.df[
            self.df.lat.between(lat_min, lat_max)
            & self.df.lon.between(lon_min, lon_max)
        ]

    def candidates_within(
        self,
        lat: float,
        lon: float,
        max_dist_m: float = DEFAULT_MATCH_GATE_M,
        k: int = 5,
    ) -> pd.DataFrame:
        """Top-k nearest poles within ``max_dist_m`` of (lat, lon).

        Adds a ``dist_m`` column. Use when nearest-neighbor is ambiguous
        (joint-use poles, dense corridors). Computed with a Cartesian
        approximation that's accurate to <0.1% inside a 10m gate at VT latitudes.
        """
        # Compute distance into a numpy array first, then filter — avoids
        # copying the full 400k-row DataFrame on every call.
        lat_m = 111_320.0
        lon_m = 111_320.0 * math.cos(math.radians(lat))
        dist_sq = (
            ((self.df.lat.values - lat) * lat_m) ** 2
            + ((self.df.lon.values - lon) * lon_m) ** 2
        )
        mask = dist_sq <= max_dist_m * max_dist_m
        if not mask.any():
            return self.df.iloc[0:0].assign(dist_m=[])  # empty, with the column shape
        sub = self.df.loc[mask].copy()
        sub["dist_m"] = dist_sq[mask] ** 0.5
        cols = ["pole_id", "utility", "natural_id", "lat", "lon", "dist_m"]
        return sub.nsmallest(k, "dist_m")[cols]

    def nearest(
        self, lat: float, lon: float, max_dist_m: float = DEFAULT_MATCH_GATE_M
    ) -> tuple[str, float] | None:
        """Single nearest pole within the gate. Returns ``(pole_id, dist_m)`` or None.

        For the typical Phase 3 flow ("match a detection or fall back"), prefer
        :meth:`match_or_provisional` — it composes this with the geohash fallback.
        """
        candidates = self.candidates_within(lat, lon, max_dist_m, k=1)
        if candidates.empty:
            return None
        row = candidates.iloc[0]
        return str(row.pole_id), float(row.dist_m)

    # ----- Phase 3 contract -----

    @staticmethod
    def provisional_id(
        lat: float, lon: float, precision: int = DEFAULT_GEOHASH_PRECISION
    ) -> str:
        """Synthetic pole_id for detections with no catalog match.

        Format ``provisional:{geohash}`` per pole_identity_design.md. Stable
        across repeat observations of the same pole as long as their GPS lands
        in the same geohash cell (~5m at precision=9).
        """
        return f"provisional:{_encode_geohash(lat, lon, precision)}"

    def match_or_provisional(
        self, lat: float, lon: float, max_dist_m: float = DEFAULT_MATCH_GATE_M
    ) -> str:
        """Return a real pole_id when a catalog pole sits within the gate;
        otherwise return ``provisional:{geohash}`` computed from the input GPS.

        This is the Phase 3 contract: every detection ends up with an ID, real
        or synthetic, and provisional IDs can be promoted later if a municipal
        utility layer becomes available.
        """
        match = self.nearest(lat, lon, max_dist_m)
        if match is not None:
            return match[0]
        return self.provisional_id(lat, lon)


if __name__ == "__main__":
    # Smoke test against the on-disk catalog. Picks an arbitrary pole, then
    # exercises each lookup pattern on it.
    cat = PoleCatalog()
    print(f"loaded {len(cat.df):,} poles from {cat.parquet_path}")

    target = cat.df.iloc[1234]
    print(f"\ntarget: {target.pole_id}  ({target.lat:.6f}, {target.lon:.6f})")

    print(f"\nfind_by_pole_id    -> {cat.find_by_pole_id(target.pole_id)['pole_id']}")
    print(f"find_by_natural_id -> {cat.find_by_natural_id(target.natural_id)}")

    # Pretend the detection drifted ~3m off the truth.
    det_lat = target.lat + 0.00003
    det_lon = target.lon - 0.00002
    nearest = cat.nearest(det_lat, det_lon)
    print(f"nearest at +3m     -> {nearest}")

    # Detection nowhere near a catalog pole — should fall back.
    print(f"match_or_provisional (off-grid) -> "
          f"{cat.match_or_provisional(45.0, -73.5)}")
