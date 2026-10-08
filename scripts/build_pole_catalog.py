"""Build the pole catalog from VAIL annotation XMLs.

For every utility-pole annotation in one route's archived XML, geohash the
labeled GPS coords and emit one row per unique pole into that route's dir.

This is the MVP catalog. `pole_id` is `geohash:<9 chars>` — derived from the
VAIL-labeled rigid coords, NOT the state utility databases. Looking up GMP /
VEC / WEC IDs is deferred (the relevant code lives in `temp_junk/` for now).
See `documentation/localization_problem.md` for the project framing.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Put the repo root on the path so local packages are importable as a script.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths  # noqa: E402
from config.paths import RoutePaths  # noqa: E402
from dataset_util.annotations_util import load_pole_rigids  # noqa: E402
from util.geohash import encode_geohash  # noqa: E402

def extract_poles(xml_path: Path) -> list[dict]:
    """Yield one row per labeled utility pole in the XML (both sides combined)."""
    poles = load_pole_rigids([xml_path], side=None)
    return [{
        "pole_id": f"geohash:{encode_geohash(p.lat, p.lon)}",
        "lat": p.lat,
        "lon": p.lon,
        "side": p.side,
        "batch": p.batch,
        "source_frame": p.filename,
    } for p in poles.itertuples()]


def load_pole_catalog(route: RoutePaths) -> pd.DataFrame:
    """Load one route's pole catalog (one row per unique pole).

    Raises FileNotFoundError if the catalog hasn't been built yet — run
    build_pole_catalog (the `build_pole_catalog` pipeline step) first.
    """
    if not route.pole_catalog.exists():
        raise FileNotFoundError(
            f"No pole catalog at {route.pole_catalog}; run build_pole_catalog first"
        )
    return pd.read_parquet(route.pole_catalog)


def build_pole_catalog(xml_files: list[Path], out_dir: Path) -> int:
    """Geohash every utility-pole annotation in `xml_files` into one deduped
    catalog (poles.parquet + poles_summary.json) under `out_dir`."""
    if not xml_files:
        print("nothing to process: no XML files given")
        return 0

    print(f"processing {len(xml_files)} XML file(s)")
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


def run_build_pole_catalog_pipeline(route: RoutePaths) -> int:
    """Pipeline entry point: catalog for ONE route, from its archived XML, into its route dir."""
    return build_pole_catalog([route.xml], route.pole_catalog.parent)


def main() -> int:
    p = argparse.ArgumentParser(description="Build one route's geohash pole catalog.")
    paths.add_batch_arg(p)
    return run_build_pole_catalog_pipeline(paths.route_paths(p.parse_args().batch))


if __name__ == "__main__":
    sys.exit(main())
