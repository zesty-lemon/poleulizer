"""Cache Grounding-DINO bbox detections for every image referenced by the
localization observations table.

Reads:    data/localization/observations.parquet
Writes:   data/detection/bboxes.parquet
            One row per detected bbox. A sentinel row with `box_index = -1`
            marks images that ran but had zero detections, so a re-run skips
            them instead of re-detecting.

Idempotent. To rebuild from scratch (e.g. after changing prompt/threshold),
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

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from detection import detection_util

STREET_IMAGES_DIR = REPO_ROOT.parent / "data" / "street_images"
OBS_PATH          = REPO_ROOT.parent / "data" / "localization" / "observations.parquet"
BBOX_OUT_PATH     = REPO_ROOT.parent / "data" / "detection" / "bboxes.parquet"

# Flush to parquet every N images so a crash doesn't lose the whole batch.
FLUSH_EVERY = 25

# Wipe bboxes.parquet before running. Use this when prompt/threshold/model/NMS
# changed and the old cache rows are stale. Leave False for normal resume runs.
CLEAR_CACHE = False


def load_cache() -> pd.DataFrame:
    """Read the existing bbox cache, or return an empty DataFrame if none exists."""
    if BBOX_OUT_PATH.exists():
        return pd.read_parquet(BBOX_OUT_PATH)
    return pd.DataFrame()


def save_cache(cache: pd.DataFrame) -> None:
    """Write the cache back to disk."""
    BBOX_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache.to_parquet(BBOX_OUT_PATH, compression="gzip", index=False)


def main() -> None:
    # 0. Optional cache wipe -- forces every image to be re-detected.
    if CLEAR_CACHE and BBOX_OUT_PATH.exists():
        BBOX_OUT_PATH.unlink()
        print(f"CLEAR_CACHE=True -- removed {BBOX_OUT_PATH}")

    # 1. Figure out which images we need detections for. The set is exactly
    #    the unique image filenames in the localization observations table.
    if not OBS_PATH.exists():
        raise FileNotFoundError(
            f"observations parquet not found at {OBS_PATH}; "
            f"run localization/localization_pipeline.py first"
        )
    obs = pd.read_parquet(OBS_PATH)
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
    n_boxes  = int((cache["box_index"] >= 0).sum())
    print(f"\nwrote {len(cache)} rows ({n_images} unique images, {n_boxes} detections)")
    print(f"  -> {BBOX_OUT_PATH}")


if __name__ == "__main__":
    main()
