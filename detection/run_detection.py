"""Cache Grounding-DINO bbox detections for every image referenced by the
localization observations table.

Reads:    data/localization/observations.parquet
Writes:   data/detection/bboxes.parquet
            One row per detected bbox. A sentinel row with `box_index = -1`
            marks images that ran but had zero detections, so a re-run skips
            them instead of re-detecting.

Important. To rebuild from scratch (e.g. after changing prompt/threshold),
delete `bboxes.parquet` first.

Run:
    python3 detection/run_detection.py

Loading downstream:
    obs    = pd.read_parquet("data/localization/observations.parquet")
    bboxes = pd.read_parquet("data/detection/bboxes.parquet")
    merged = obs.merge(bboxes, on="image_filename", how="left")
"""

import sys
from pathlib import Path

import argparse

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import paths
from config.paths import RoutePaths
from detection import detection_util

STREET_IMAGES_DIR = paths.STREET_IMAGES_DIR
BBOX_OUT_PATH = paths.BBOX_PATH

# Flush to parquet every N images so a crash doesn't lose the whole batch.
FLUSH_EVERY = 25


def load_cache() -> pd.DataFrame:
    """Read the existing bbox cache, or return an empty DataFrame if none exists.

    This is the resume-safe internal helper for the detection loop (missing file
    means "nothing cached yet, start fresh"). For retrieving finished detections,
    use load_detections(), which raises if the file is absent.
    """
    if BBOX_OUT_PATH.exists():
        return pd.read_parquet(BBOX_OUT_PATH)
    return pd.DataFrame()


def load_detections() -> pd.DataFrame:
    """Load the detection bounding boxes (one row per box; box_index == -1 is a
    zero-detection sentinel row). Key `image_filename` joins to observations.

    Raises FileNotFoundError if detection hasn't been run yet — run
    detect_poles (the `detect_poles` pipeline step) first.
    """
    if not BBOX_OUT_PATH.exists():
        raise FileNotFoundError(
            f"No detections at {BBOX_OUT_PATH}; run detect_poles first"
        )
    return pd.read_parquet(BBOX_OUT_PATH)


def save_cache(cache: pd.DataFrame) -> None:
    """Write the cache back to disk."""
    BBOX_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(BBOX_OUT_PATH, compression="gzip", index=False)


def run_detection_and_save(route: RoutePaths, clear_cache: bool = False):
    """
    Put bounding boxes around all images located by the localization_pipeline.py
    Assumes localization_pipeline.py has already been run. Bounding Boxes are saved
    to bboxes.parquet file.

    :param clear_cache: # Wipe bboxes.parquet before running. Use this when prompt/threshold/model/NMS
    changed and the old cache rows are stale. Leave False for normal resume runs.
    :param route: which route's observations decide the image set. The bbox cache
    itself is shared across routes (keyed by image filename), so it is never route-scoped.

    :return:
    """
    obs_path = route.observations

    # 0. Optional cache wipe -- forces every image to be re-detected.
    if clear_cache and BBOX_OUT_PATH.exists():
        BBOX_OUT_PATH.unlink()
        print(f"clear_cache=True -- removed {BBOX_OUT_PATH}")

    # 1. Figure out which images we need detections for. The set is exactly
    #    the unique image filenames in the localization observations table.
    if not obs_path.exists():
        raise FileNotFoundError(
            f"observations parquet not found at {obs_path}; "
            f"run localization/localization_pipeline.py first"
        )
    obs = pd.read_parquet(obs_path)
    all_images = sorted(obs["image_filename"].dropna().unique())

    # 2. Skip anything already cached. An image is "done" if its filename
    #    appears in the cache at all (detection sentinel or real boxes).
    cache = load_cache()
    already_done = set(cache["image_filename"]) if not cache.empty else set()
    todo = [name for name in all_images if name not in already_done]

    print(f"observations: {len(obs)} rows / {len(all_images)} unique images")
    print(f"cache:        {len(already_done)} done -> {len(todo)} remaining")
    if not todo:
        print("nothing to do.")
        return

    # 3. Load the model once, then run detection on every remaining image.
    print(f"loading {detection_util.MODEL_ID} on {detection_util.DEFAULT_DEVICE}")
    processor, model = detection_util.load_detector()

    pending: list[dict] = []
    for i, fname in enumerate(tqdm(todo, desc="detecting"), start=1):
        image_path = STREET_IMAGES_DIR / fname
        if not image_path.is_file():
            print(f"  WARN: missing image {image_path}, skipping")
            continue

        pending.extend(detection_util.detect_image(image_path, processor, model))

        if i % FLUSH_EVERY == 0:
            cache = pd.concat([cache, pd.DataFrame(pending)], ignore_index=True)
            save_cache(cache)
            pending = []

    # 4. Flush whatever didn't land on a FLUSH_EVERY boundary.
    if pending:
        cache = pd.concat([cache, pd.DataFrame(pending)], ignore_index=True)
        save_cache(cache)

    n_images = cache["image_filename"].nunique()
    n_boxes = int((cache["box_index"] >= 0).sum())
    print(f"\nwrote {len(cache)} rows ({n_images} unique images, {n_boxes} detections)")
    print(f"  -> {BBOX_OUT_PATH}")


def main() -> None:
    p = argparse.ArgumentParser(description="Detect pole bounding boxes for one route's frames.")
    paths.add_batch_arg(p)
    p.add_argument("--clear-cache", action="store_true",
                   help="wipe the SHARED bbox cache first (renumbers box_index; voids ground truth)")
    args = p.parse_args()
    run_detection_and_save(paths.route_paths(args.batch), clear_cache=args.clear_cache)


if __name__ == "__main__":
    main()
