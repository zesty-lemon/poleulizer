"""Centralized filesystem paths for the project.

All generated data lives OUTSIDE the repo, one level up, in ../data/, so the
repo stays code-only. Every data path the pipeline reads or writes is derived
here, so the layout lives in exactly one place instead of being re-derived in
each script.
"""
from dataclasses import dataclass
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

# Detection bbox cache.
DETECTION_DIR = DATA_DIR / "detection"
BBOX_PATH = DETECTION_DIR / "bboxes.parquet"

# ----- Per-route outputs -----
#
# One directory per batch (= XML stem); everything specific to a route lives here.
# Shared caches (street_images/, detection/) stay at the top level because they
# are keyed by image filename and are reused across routes.
ROUTES_DIR = DATA_DIR / "routes"


@dataclass(frozen=True)
class RoutePaths:
    """All paths for one onboarded route, derived from its batch name.

    The batch name is used verbatim as the directory name so it matches the
    archived XML's stem and the `batch` column in observations/matches exactly.
    """
    batch: str

    @property
    def root(self) -> Path:
        return ROUTES_DIR / self.batch

    # ----- input
    @property
    def xml(self) -> Path:
        """The archived annotation XML this route was built from."""
        return ONBOARDED_ROUTES_DIR / f"{self.batch}.xml"

    # ----- pipeline artifacts
    @property
    def pole_catalog(self) -> Path:
        return self.root / "pole_catalog" / "poles.parquet"

    @property
    def pole_catalog_summary(self) -> Path:
        return self.root / "pole_catalog" / "poles_summary.json"

    @property
    def observations(self) -> Path:
        return self.root / "localization" / "observations.parquet"

    @property
    def matches(self) -> Path:
        return self.root / "matching" / "pole_box_matches.parquet"

    @property
    def isolated_poles(self) -> Path:
        """(image_filename, pole_id) subset where the pole is alone in its frame."""
        return self.root / "matching" / "isolated_poles.csv"

    # ----- evaluation (the hand annotation lives alongside what it scores)
    @property
    def ground_truth_template(self) -> Path:
        return self.root / "evaluation" / "ground_truth_template.csv"

    @property
    def ground_truth(self) -> Path:
        return self.root / "evaluation" / "ground_truth.csv"

    @property
    def eval_reports_dir(self) -> Path:
        return self.root / "evaluation" / "reports"

    # ----- deliverables
    @property
    def export_dir(self) -> Path:
        return self.root / "pole_export"

    @property
    def complete_route_export_dir(self) -> Path:
        return self.export_dir / "complete_route"

    @property
    def isolated_export_dir(self) -> Path:
        return self.export_dir / "isolated_poles"


def route_paths(batch: str) -> RoutePaths:
    return RoutePaths(batch)


def add_batch_arg(parser) -> None:
    """Add the standard `--batch` CLI option (route = XML stem) to an argparse parser."""
    parser.add_argument("--batch", required=True, metavar="NAME",
                        help="route to operate on: the batch name (= annotation XML stem)")
