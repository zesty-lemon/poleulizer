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

from config import paths  # noqa: E402

DEFAULT_ARCHIVE = paths.ONBOARDED_ROUTES_DIR

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


def _read_frames(xml_path: Path) -> list[dict]:
    """One row per <annotation> (= one frame): where the camera was and which JPGs it has.

    Schema (VAIL export, current): <sources>/src_{center,left,right}, <location>/{latitude,
    longitude,heading}. Rigid labels live in a separate table, see _read_rigids.
    """
    rows: list[dict] = []
    root = ET.parse(xml_path).getroot()
    batch = xml_path.stem
    for ann in root.findall("annotation"):
        rows.append({
            "batch": batch,
            "filename": ann.findtext("filename"),
            "src_center": _grab_filename(ann.findtext("sources/src_center")),
            "src_left": _grab_filename(ann.findtext("sources/src_left")),
            "src_right": _grab_filename(ann.findtext("sources/src_right")),
            "source_lat": _to_float(ann.findtext("location/latitude")),
            "source_lon": _to_float(ann.findtext("location/longitude")),
            "source_heading_xml": _to_float(ann.findtext("location/heading")),
        })
    return rows


def _read_rigids(xml_path: Path) -> list[dict]:
    """One row per labeled rigid object: <sides>/<side>/rigids/rigid (0..n per side).

    Rows without a usable GPS fix are dropped. `filename` is the frame the label
    was placed on; join to the frames table on (batch, filename).
    """
    rows: list[dict] = []
    root = ET.parse(xml_path).getroot()
    batch = xml_path.stem
    for ann in root.findall("annotation"):
        filename = ann.findtext("filename")
        for side in ("right", "left"):
            for rigid in ann.findall(f"sides/{side}/rigids/rigid"):
                lat = _to_float(rigid.findtext("gps/latitude"))
                lon = _to_float(rigid.findtext("gps/longitude"))
                if lat is None or lon is None:
                    continue
                rows.append({
                    "batch": batch,
                    "filename": filename,
                    "side": side,
                    "rigid_type": _clean(rigid.findtext("type")),
                    "lat": lat,
                    "lon": lon,
                })
    return rows


def _frame_sort_key(s: pd.Series) -> pd.Series:
    # Composite sort key: session timestamp (suffix) then frame counter (prefix).
    # Both are fixed-width zero-padded, so lex sort = chronological + route order.
    return s.str.split("_").str[-1] + s.str.split("_").str[0]


def load_annotations(xml_paths: list[Path]) -> pd.DataFrame:
    """Read annotation XML(s) into a frame table: one row per frame, in route order
    (earliest capture first)."""
    rows: list[dict] = []
    for xml_path in xml_paths:
        rows.extend(_read_frames(Path(xml_path)))
    df = pd.DataFrame(rows)
    return df.sort_values("filename", key=_frame_sort_key).reset_index(drop=True)


def load_rigids(xml_paths: list[Path]) -> pd.DataFrame:
    """Read annotation XML(s) into a rigid table: one row per labeled rigid object,
    in the same route order as load_annotations."""
    rows: list[dict] = []
    for xml_path in xml_paths:
        rows.extend(_read_rigids(Path(xml_path)))
    df = pd.DataFrame(rows, columns=["batch", "filename", "side", "rigid_type", "lat", "lon"])
    return df.sort_values("filename", key=_frame_sort_key, kind="stable").reset_index(drop=True)


def load_pole_rigids(xml_paths: list[Path], side: str | None = "right") -> pd.DataFrame:
    """Rigid rows that are utility poles; side='right' (default), 'left', or None for both.

    Left-side poles are excluded by default for now — pass side=None when the
    left side comes into scope.
    """
    rigids = load_rigids(xml_paths)
    poles = rigids[rigids.rigid_type == str(c.UTILITY_POLE)]
    return poles if side is None else poles[poles.side == side]


if __name__ == "__main__":
    import sys
    xmls = [Path(p) for p in sys.argv[1:]] or sorted(DEFAULT_ARCHIVE.glob("*.xml"))
    frames, rigids = load_annotations(xmls), load_rigids(xmls)
    print(f"{len(frames)} frames, {len(rigids)} rigids, {len(load_pole_rigids(xmls))} right-side poles")
    print(frames.head()); print(rigids.head())

