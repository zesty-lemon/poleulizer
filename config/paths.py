"""Centralized filesystem paths for the project.

All generated data lives OUTSIDE the repo, one level up, in ../data/, so the
repo stays code-only. Every data path the pipeline reads or writes is derived
here, so the layout lives in exactly one place instead of being re-derived in
each script.
"""
from pathlib import Path

# Repo root = the parent of the config/ package this file lives in.
REPO_ROOT = Path(__file__).resolve().parent.parent

# All artifacts live one level above the repo, in ../data/.
DATA_DIR = REPO_ROOT.parent / "data"

# Route annotation XMLs: inbox -> archive workflow.
ROUTE_ANNOTATIONS_DIR = DATA_DIR / "route_annotations"
NEW_ROUTES_DIR = ROUTE_ANNOTATIONS_DIR / "new_routes"              # inbox: drop XMLs here
ONBOARDED_ROUTES_DIR = ROUTE_ANNOTATIONS_DIR / "onboarded_routes"  # archive: fully-fetched XMLs

# Downloaded POV JPGs (flat layout).
STREET_IMAGES_DIR = DATA_DIR / "street_images"

# Geohash pole catalog.
POLE_CATALOG_DIR = DATA_DIR / "pole_catalog"
POLE_CATALOG_PATH = POLE_CATALOG_DIR / "poles.parquet"
POLE_CATALOG_SUMMARY_PATH = POLE_CATALOG_DIR / "poles_summary.json"

# Localization observations (which frames show each pole).
LOCALIZATION_DIR = DATA_DIR / "localization"
OBSERVATIONS_PATH = LOCALIZATION_DIR / "observations.parquet"

# Detection bbox cache.
DETECTION_DIR = DATA_DIR / "detection"
BBOX_PATH = DETECTION_DIR / "bboxes.parquet"
