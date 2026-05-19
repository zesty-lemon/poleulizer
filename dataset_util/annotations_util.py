"""Read VAIL annotation XMLs from onboarded_routes/ into a flat DataFrame.

Returns columns from annotation xml: frame ID, three view URLs, source GPS, and
the per-side rigid-obstacle labels.

"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Make `config` importable when this file is executed directly as a script.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

import config.constants as c  # noqa: E402
import pandas as pd  # noqa: E402

DEFAULT_ARCHIVE = REPO_ROOT.parent / "data" / "route_annotations" / "onboarded_routes"

def _clean(text: str | None) -> str | None:
    """Treat the literal string 'None' (the schema's null sentinel) as a real None."""
    if text is None:
        return None
    s = text.strip()
    return None if s == "" or s.lower() == "none" else s


def _to_float(text: str | None) -> float | None:
    s = _clean(text)
    return float(s) if s is not None else None


def _grab_filename(image_path: str) -> str:
    if image_path == "" or image_path == None:
        return ""
    else:
        return image_path.rsplit('/', 1)[1]


def _read_one(xml_path: Path) -> list[dict]:
    """Yield one row dict per <annotation> in the XML."""
    rows: list[dict] = []
    root = ET.parse(xml_path).getroot()
    batch = xml_path.stem
    for ann in root.findall("annotation"):
        row = {
            "batch": batch,
            "filename": ann.findtext("filename"),
            "src_center": _grab_filename(ann.findtext("source/src_center")),
            "src_left": _grab_filename(ann.findtext("source/src_left")),
            "src_right": _grab_filename(ann.findtext("source/src_right")),
            "source_lat": _to_float(ann.findtext("location/latitude")),
            "source_lon": _to_float(ann.findtext("location/longitude")),
        }
        for side in ("right", "left"):
            row[f"{side}_rigid_type"] = _clean(ann.findtext(f"tags/{side}/rigid/type"))
            row[f"{side}_rigid_lat"] = _to_float(ann.findtext(f"tags/{side}/rigid/gps/latitude"))
            row[f"{side}_rigid_lon"] = _to_float(ann.findtext(f"tags/{side}/rigid/gps/longitude"))
        rows.append(row)
    return rows


def load_annotations(xml_paths: list[Path] | None = None) -> pd.DataFrame:
    """Read one or more annotation XMLs into a flat DataFrame.

    Defaults to a single hardcoded XML from the onboarded archive — replace
    with the full glob (`sorted(DEFAULT_ARCHIVE.glob('*.xml'))`) when you want
    every onboarded route at once.

    Returns in order present in the original XML file

    The top of the dataframe is the EARLIEST photos, the bottom is the LATEST
    """
    if xml_paths is None:
        xml_paths = [
            DEFAULT_ARCHIVE
            / "Batch-5-US_5_St_Johnsbury_(Sameeh)(Ready_for_VTrans_Review)_merged.xml"
        ]
    rows: list[dict] = []

    # read all xml files in xml_paths into our dataframe
    for xml_path in xml_paths:
        rows.extend(_read_one(Path(xml_path)))
    df = pd.DataFrame(rows)
    # Composite sort key: session timestamp (suffix) then frame counter (prefix).
    # Both are fixed-width zero-padded, so lex sort = chronological + route order.
    return df.sort_values(
        "filename",
        key=lambda s: s.str.split("_").str[-1] + s.str.split("_").str[0],
    ).reset_index(drop=True)


def load_right_pole_annotations(xml_paths: list[Path] | None = None) -> pd.DataFrame:
    """Annotations whose RIGHT-side rigid label is a utility pole.

    Left-side poles are intentionally excluded for now — extend this when the
    left side comes into scope.
    """
    all_annotations = load_annotations(xml_paths)
    pole_df = all_annotations[all_annotations['right_rigid_type'] == str(c.UTILITY_POLE)]
    return pole_df



if __name__ == "__main__":
    # df = load_right_pole_annotations()
    # print(f"loaded {len(df)} annotations from {df.batch.nunique()} batch(es)")
    # print(df.head())
    # rigid = df[df.right_rigid_type.notna() | df.left_rigid_type.notna()]
    # print(f"\n{len(rigid)} annotations with a rigid-class label on either side")

    all_annotations = load_annotations()
    print(all_annotations.head())

