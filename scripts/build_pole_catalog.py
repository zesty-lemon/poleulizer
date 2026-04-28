#!/usr/bin/env python3
"""Build the unified Vermont pole catalog from three VCGI utility layers.

Phase 3 of the project (detection -> pole match) needs a queryable table of
every known Vermont utility pole with a stable primary key. This script
produces that table.

In Vermont, Utility poles are indexed and mapped bu the three main utilities
These maps contain pole-> ID references and do not overlap with each other.

Business logic:
  1. Paginate through three public ArcGIS endpoints (GMP + VEC FeatureServers,
     WEC MapServer) until each layer is exhausted. Someonetimes the server sends
     fewer records than we request (it has a smaller limit), so we take this into account
     and only request the max number it will actually give us.
  2. We do not need to adjust the chart datym. The server reprojects features to WGS84 via outSR=4326,
     this avoids a pyproj dependency/conversion and yields lat/lon directly comparable to dashcam GPS.
  3. Extract the per-utility natural key (POLETAG for GMP, gs_guid for VEC and
     WEC). Rows with no key are dropped since we can't reference an unnamed pole.
  4. Build pole_id = "{utility}:{natural_id}"
     This is the catalog's primary key; the identity scheme requires global
     uniqueness, so the script fails loud on any collision rather than write a
     broken catalog.
  5. Surface joint_use only for GMP (TELPOLE/JCTPOLE flags); VEC and WEC have
     no equivalent so they leave it null. The full original attribute set is
     preserved as a JSON blob in source_attrs for forensic lookup.

Failure handling: per-page retries on 5xx and connection errors with (2,4,8)s
backoff; 4xx is treated as permanent. A layer-level failure is recorded in the
summary but does not abort other layers. Exit code is non-zero if anything
failed or if pole_id uniqueness was violated.

Outputs:
  data/pole_catalog/poles.parquet      -- ~399k rows, gzip-compressed
  data/pole_catalog/poles_summary.json -- per-utility counts + fetch timestamp

The catalog is rebuilt from scratch every run, so only run when needed to update the catalog, not every time
the application restarts.

Reference:
General Guide to Utility Poles & Schema for Vermont. All utilities are supposed to follow this standard.
https://vcgi.vermont.gov/sites/vcgiupdate/files/doc_library/VTPoleData%20GIS%20Standard%20v1_20201208.pdf.pdf

General Sources/Reading:
  - https://geodata.vermont.gov/datasets/VCGI::vt-green-mountain-power-pole-data/about
  - https://maps.vcgi.vermont.gov/gisdata/metadata/UtilityTransmit_GMPPOLES.htm
  - https://geodata.vermont.gov/datasets/VCGI::vt-data-vec-poles-points/about
  - https://geodata.vermont.gov/datasets/vtpsd::vt-data-wec-utility-poles
  - https://vcgi.vermont.gov/document/vermont-utility-pole-gis-data-standard
  - https://vcgi.vermont.gov/sites/vcgiupdate/files/doc_library/VTPoleData%20GIS%20Standard%20v1_20201208.pdf.pdf
  - https://geodata.vermont.gov/

Limitations:
Coverage does not include municipal utilities (Burlington Electric, Morrisville Water & Light, Stowe Electric, Hardwick Electric, Hyde Park, etc.)
This is why we have a hash lookup for poles that are not present in our 3 main data layers. This does not happen in this file.
If we try and lookup a pole here that does not exist in this database, we hash the lat/long and then insert it into
the database as its ID. This should be a rare fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# Paths resolve from the script's own location so it works from any CWD.
# Workaround for always hardoding abasolute paths
# Todo: move to utility/constants?
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT.parent / "data"
DEFAULT_OUT = DATA_DIR / "pole_catalog"

# ArcGIS Connection Settings
PAGE_SIZE = 2000
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 60
BACKOFF_SECONDS = (2, 4, 8)

# similar to defining a custom object in java
@dataclass(frozen=True)
class LayerSpec:
    """One VCGI-hosted utility pole layer we intend to merge into the catalog."""
    utility: str            # "GMP" | "VEC" | "WEC"
    url: str                # ArcGIS query endpoint (ends in /query)
    natural_key_field: str  # server-side field name holding the stable ID


# Three per-utility layers, keyed by a short argparse-friendly name.
LAYERS: dict[str, LayerSpec] = {
    "gmp": LayerSpec(
        utility="GMP",
        url=(
            "https://services1.arcgis.com/BkFxaEFNwHqX3tAw/arcgis/rest/services/"
            "FS_VCGI_OPENDATA_Utility_GMP_Poles_point_SP_v1/FeatureServer/0/query"
        ),
        natural_key_field="POLETAG",
    ),
    "vec": LayerSpec(
        utility="VEC",
        url=(
            "https://services1.arcgis.com/BkFxaEFNwHqX3tAw/arcgis/rest/services/"
            "FS_VCGI_OPENDATA_Utility_VEC_POLES_point_SP_v1/FeatureServer/0/query"
        ),
        natural_key_field="gs_guid",
    ),
    "wec": LayerSpec(
        utility="WEC",
        url=(
            "https://maps.vcgi.vermont.gov/arcgis/rest/services/PSD_services/"
            "OPENDATA_PSD_LAYERS_SP_NOCACHE_v1/MapServer/35/query"
        ),
        natural_key_field="gs_guid",
    ),
}


def build_session() -> requests.Session:
    """
    Builds the https session request
    :return: session to access GIS layers
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "poleulizer-catalog/0.1"})
    return session


def fetch_page(
    session: requests.Session, url: str, offset: int, page_size: int
) -> dict[str, Any]:
    """Fetch one paginated page of features. Retries on 5xx / connection errors.

    `outSR=4326` asks the server to reproject to WGS84 so we don't have to pull
    in pyproj. `f=geojson` gives us the standard GeoJSON FeatureCollection
    shape. A 4xx response or an ArcGIS-level `{"error": ...}` body is treated
    as permanent and bubbles up immediately.
    """
    params = {
        "where": "1=1",
        "outFields": "*",
        "outSR": "4326",
        "f": "geojson",
        "resultOffset": offset,
        "resultRecordCount": page_size,
    }
    last_exc: Exception | None = None
    for attempt in range(len(BACKOFF_SECONDS) + 1):
        try:
            response = session.get(
                url, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )
            # 4xx is not retried, 4xx requests should fail
            if 400 <= response.status_code < 500:
                raise RuntimeError(
                    f"client error {response.status_code}: {response.text[:300]}"
                )
            response.raise_for_status()  # now catches 5xx
            body = response.json()
            # ArcGIS sometimes returns HTTP 200 with an error payload for NO REASON
            # catch these gracefully. Blame project managers not understanding error codes
            if isinstance(body, dict) and "error" in body:
                raise RuntimeError(f"ArcGIS error body: {body['error']}")
            return body
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            if attempt < len(BACKOFF_SECONDS):
                time.sleep(BACKOFF_SECONDS[attempt])
                continue
            raise
    # Defensive — loop above either returns or raises.
    raise RuntimeError(f"unreachable; last exc: {last_exc}")


def fetch_count(session: requests.Session, url: str) -> int:
    """Ask the server for total feature count (fast; used by --dry-run)."""
    params = {"where": "1=1", "returnCountOnly": "true", "f": "json"}
    response = session.get(
        url, params=params, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
    )
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"ArcGIS error body: {body['error']}")
    return int(body.get("count", 0))


def extract_row(
    feature: dict[str, Any], spec: LayerSpec, fetched_at: str, source_base_url: str
) -> dict[str, Any] | None:
    """Turn one GeoJSON feature into a catalog row. Returns None to drop it."""
    attrs = feature.get("properties") or {}
    raw_key = attrs.get(spec.natural_key_field)
    if raw_key is None or str(raw_key).strip() == "":
        return None
    natural_id = str(raw_key).strip()

    # GeoJSON geometry.coordinates is [lon, lat].
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates") or [None, None]
    lon, lat = coords[0], coords[1]

    # Utility-specific fields: GMP has TELPOLE/JCTPOLE joint-use flags; VEC/WEC
    # don't expose an equivalent. Owner comes from different columns per layer.
    if spec.utility == "GMP":
        tel_flag = attrs.get("TELPOLE")
        jct_flag = attrs.get("JCTPOLE")
        if tel_flag is None and jct_flag is None:
            joint_use: bool | None = None
        else:
            joint_use = bool(tel_flag) or bool(jct_flag)
        owner = attrs.get("OWNER")
    else:
        joint_use = None
        owner = attrs.get("gs_owner")

    return {
        "pole_id": f"{spec.utility}:{natural_id}",
        "utility": spec.utility,
        "natural_id": natural_id,
        "lat": lat,
        "lon": lon,
        "owner": owner,
        "joint_use": joint_use,
        # default=str handles datetimes and any other non-JSON-native types
        # ArcGIS might serialize into the attributes dict.
        "source_attrs": json.dumps(attrs, default=str),
        "source_layer": source_base_url,
        "fetched_at": fetched_at,
    }


def fetch_layer(
    session: requests.Session, spec: LayerSpec, page_size: int, fetched_at: str
) -> tuple[list[dict[str, Any]], int]:
    """Paginate through one layer; return (rows, dropped_count)."""
    # Strip the trailing "/query" for the provenance column — that URL is what
    # the server uses to expose metadata pages, friendlier in the manifest.
    base_url = (
        spec.url.removesuffix("/query") if spec.url.endswith("/query") else spec.url
    )

    rows: list[dict[str, Any]] = []
    dropped = 0
    offset = 0
    while True:
        print(f"  [{spec.utility}] fetching offset={offset}...", flush=True)
        body = fetch_page(session, spec.url, offset, page_size)
        features = body.get("features") or []
        if not features:
            break
        for feat in features:
            row = extract_row(feat, spec, fetched_at, base_url)
            if row is None:
                dropped += 1
            else:
                rows.append(row)
        # Advance by what the server actually returned, not by what we asked
        # for: MapServer endpoints may cap a page below page_size while still
        # having more rows. The empty-page check above is the real terminator.
        offset += len(features)
    return rows, dropped


def parse_utilities(raw: str) -> tuple[str, ...]:
    requested = tuple(u.strip().lower() for u in raw.split(",") if u.strip())
    unknown = [u for u in requested if u not in LAYERS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown utility/ies: {unknown}. allowed: {tuple(LAYERS.keys())}"
        )
    return requested or tuple(LAYERS.keys())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch VT utility pole GIS layers and union into poles.parquet."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--util",
        type=parse_utilities,
        default=tuple(LAYERS.keys()),
        help=f"comma-separated subset of {tuple(LAYERS.keys())} (default: all)",
    )
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out.expanduser().resolve()
    fetched_at = datetime.now(timezone.utc).isoformat()
    session = build_session()

    # --dry-run just asks the server for a row count per layer.
    # Exists to check the endpoints are correct and everything is working
    if args.dry_run:
        print("dry-run: querying feature counts")
        any_failed = False
        for utility_key in args.utilities:
            spec = LAYERS[utility_key]
            try:
                count = fetch_count(session, spec.url)
                print(f"  [{spec.utility}] {count} features  ({spec.url})")
            except Exception as exc:
                print(f"  [{spec.utility}] FAILED: {exc}", file=sys.stderr)
                any_failed = True
        return 1 if any_failed else 0

    # Full fetch. Per-utility failures don't abort the whole run; they're
    # recorded in the summary and surface via non-zero exit at the end.
    all_rows: list[dict[str, Any]] = []
    per_utility_summary: dict[str, dict[str, Any]] = {}
    any_failed = False

    for utility_key in args.utilities:
        spec = LAYERS[utility_key]
        print(f"[{spec.utility}] starting ({spec.url})")
        try:
            rows, dropped = fetch_layer(session, spec, args.page_size, fetched_at)
        except Exception as exc:
            print(f"[{spec.utility}] FAILED: {exc}", file=sys.stderr)
            per_utility_summary[spec.utility] = {
                "url": spec.url,
                "status": "failed",
                "error": str(exc),
            }
            any_failed = True
            continue
        per_utility_summary[spec.utility] = {
            "url": spec.url,
            "status": "ok",
            "row_count": len(rows),
            "dropped_no_key": dropped,
        }
        print(f"[{spec.utility}] {len(rows)} rows (dropped {dropped} with empty key)")
        all_rows.extend(rows)

    if not all_rows:
        print("error: no rows fetched from any layer", file=sys.stderr)
        return 1

    df = pd.DataFrame(all_rows)

    # Uniqueness is important. If there is a pole_id collision we want to fail with an error
    duplicated = df[df["pole_id"].duplicated(keep=False)]
    if not duplicated.empty:
        print(
            f"error: {len(duplicated)} rows share a pole_id. Sample:",
            file=sys.stderr,
        )
        print(duplicated.head(10).to_string(), file=sys.stderr)
        return 1

    # Write summary & parquet file to disc
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "poles.parquet"
    df.to_parquet(parquet_path, compression="gzip", index=False)

    summary = {
        "fetched_at": fetched_at,
        "total_rows": int(len(df)),
        "by_utility": per_utility_summary,
    }
    summary_path = out_dir / "poles_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"wrote {parquet_path} ({len(df)} rows, {size_mb:.1f} MB)")
    print(f"wrote {summary_path}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
