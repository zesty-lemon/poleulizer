"""Build the pole catalog from VAIL annotation XMLs.

For every utility-pole annotation in `data/route_annotations/onboarded_routes/`,
geohash the labeled GPS coords and emit one row per unique pole.

This is the MVP catalog. `pole_id` is `geohash:<9 chars>` — derived from the
VAIL-labeled rigid coords, NOT the state utility databases. Looking up GMP /
VEC / WEC IDs is deferred (the relevant code lives in `temp_junk/` for now).
See `documentation/localization_problem.md` for the project framing.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Put the repo root on the path so local packages are importable as a script.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from config import constants  # noqa: E402
from config import paths  # noqa: E402
from dataset_util.annotations_util import _clean, _to_float  # noqa: E402
from util.geohash import encode_geohash  # noqa: E402

# Catalog is built from the onboarded (fully-downloaded) routes.
DEFAULT_INBOX = paths.ONBOARDED_ROUTES_DIR
DEFAULT_OUT = paths.POLE_CATALOG_DIR

UTILITY_POLE_CODE = str(constants.UTILITY_POLE)


def extract_poles(xml_path: Path) -> list[dict]:
    """Yield one row per labeled utility pole in the XML (both sides combined)."""
    rows: list[dict] = []
    batch = xml_path.stem
    root = ET.parse(xml_path).getroot()
    for ann in root.findall("annotation"):
        filename = ann.findtext("filename") or ""
        for side in ("right", "left"):
            rigid_type = _clean(ann.findtext(f"tags/{side}/rigid/type"))
            if rigid_type != UTILITY_POLE_CODE:
                continue
            lat = _to_float(ann.findtext(f"tags/{side}/rigid/gps/latitude"))
            lon = _to_float(ann.findtext(f"tags/{side}/rigid/gps/longitude"))
            if lat is None or lon is None:
                continue
            rows.append({
                "pole_id": f"geohash:{encode_geohash(lat, lon)}",
                "lat": lat,
                "lon": lon,
                "side": side,
                "batch": batch,
                "source_frame": filename,
            })
    return rows


def load_pole_catalog() -> pd.DataFrame:
    """Load the pole catalog (one row per unique pole).

    Raises FileNotFoundError if the catalog hasn't been built yet — run
    build_pole_catalog (the `build_pole_catalog` pipeline step) first.
    """
    if not paths.POLE_CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"No pole catalog at {paths.POLE_CATALOG_PATH}; "
            f"run build_pole_catalog first"
        )
    return pd.read_parquet(paths.POLE_CATALOG_PATH)


def main() -> int:
    inbox: Path = DEFAULT_INBOX
    out_dir: Path = DEFAULT_OUT

    if not inbox.is_dir():
        print(f"error: onboarded-routes dir not found at {inbox}", file=sys.stderr)
        return 2
    xml_files = sorted(inbox.glob("*.xml"))
    if not xml_files:
        print(f"nothing to process: no XML files in {inbox}")
        return 0

    print(f"processing {len(xml_files)} XML file(s) from {inbox.name}/")
    all_rows: list[dict] = []
    for xml_path in xml_files:
        rows = extract_poles(xml_path)
        print(f"  [{xml_path.name}] {len(rows)} utility-pole annotations")
        all_rows.extend(rows)

    if not all_rows:
        print("no utility-pole annotations found; nothing to write")
        return 0

    df = pd.DataFrame(all_rows)
    # Two annotations of the same physical pole (across batches, across re-survey
    # passes) land in the same ~5m geohash cell and collapse to one catalog row.
    before = len(df)
    df = df.drop_duplicates(subset="pole_id", keep="first").reset_index(drop=True)
    collapsed = before - len(df)
    if collapsed > 0:
        print(f"  collapsed {collapsed} duplicate observation(s) by geohash cell")

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "poles.parquet"
    df.to_parquet(parquet_path, compression="gzip", index=False)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "total_poles": int(len(df)),
        "duplicates_collapsed": int(collapsed),
        "source_xml_files": [p.name for p in xml_files],
    }
    summary_path = out_dir / "poles_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    size_kb = parquet_path.stat().st_size / 1024
    print(f"\nwrote {parquet_path} ({len(df)} poles, {size_kb:.1f} KB)")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
