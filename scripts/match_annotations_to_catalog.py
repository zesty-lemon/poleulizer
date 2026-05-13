"""Match labeled utility-pole annotations to canonical catalog IDs.

Phase 3 starter: walks the annotations DataFrame, picks rows where rigid.type
is the utility-pole code, and calls PoleCatalog.match_or_provisional() on each
pole's GPS to get a stable pole_id (real or provisional).

Each annotation can carry up to two poles (right side + left side), so the
inner loop iterates over both sides independently.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly from any CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import constants
from dataset_util.annotations_util import load_annotations
from dataset_util.pole_catalog_util import PoleCatalog


def main() -> int:
    df = load_annotations()
    catalog = PoleCatalog()

    # The DataFrame stores rigid_type as a string (e.g. "2") because that's
    # what comes out of the XML; constants.UTILITY_POLE is the int 2.
    utility_pole_code = str(constants.UTILITY_POLE)

    matches: list[tuple[str, str, float, float, str]] = []
    matched_count = 0
    provisional_count = 0

    for ann in df.itertuples(index=False):
        for side in ("right", "left"):
            if getattr(ann, f"{side}_rigid_type") != utility_pole_code:
                continue
            lat = getattr(ann, f"{side}_rigid_lat")
            lon = getattr(ann, f"{side}_rigid_lon")
            if lat is None or lon is None:
                continue

            pole_id = catalog.match_or_provisional(lat, lon)

            matches.append((ann.filename, side, lat, lon, pole_id))
            if pole_id.startswith("provisional:"):
                provisional_count += 1
            else:
                matched_count += 1

    print(f"processed {len(df)} annotations")
    print(f"  matched to catalog:        {matched_count}")
    print(f"  fell back to provisional:  {provisional_count}")
    print(f"  total utility poles:       {len(matches)}")

    print("\nfirst 5 results:")
    for filename, side, lat, lon, pole_id in matches[:5]:
        print(f"  {filename[:40]:40}  {side:>5}  ({lat:.6f},{lon:.6f}) -> {pole_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
