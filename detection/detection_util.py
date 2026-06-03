"""Grounding-DINO zero-shot bounding-box detection for utility poles.

Thin wrapper around `IDEA-Research/grounding-dino-base` from `transformers`:
- `load_detector()` builds processor+model once
- `detect_image()` runs one image through the pipeline and returns a list of
  row-dicts (one per detected box) ready to concat into a DataFrame

Forced to CPU until we have a stable baseline; flip `DEFAULT_DEVICE` to "mps"
when needed.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_PROMPT = "utility pole."
DEFAULT_DEVICE = "mps"
DEFAULT_THRESHOLD = 0.25
DEFAULT_TEXT_THRESHOLD = 0.25
# IoU threshold for non-max suppression — Grounding-DINO often stacks several
# overlapping boxes on the same pole. Boxes with IoU > this are collapsed to
# the highest-scoring one. Lower = more aggressive deduping.
DEFAULT_NMS_IOU = 0.5


def load_detector(device: str = DEFAULT_DEVICE):
    """Build (processor, model) for grounding-DINO. Call once, reuse forever."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
    model.eval()
    return processor, model


def detect_image(
    image_path: Path,
    processor,
    model,
    device: str = DEFAULT_DEVICE,
    prompt: str = DEFAULT_PROMPT,
    threshold: float = DEFAULT_THRESHOLD,
    text_threshold: float = DEFAULT_TEXT_THRESHOLD,
    nms_iou: float = DEFAULT_NMS_IOU,
) -> list[dict]:
    """Run the detector on a single JPG. Return one dict per detected box.

    A processed-but-empty image returns a single sentinel row with
    `box_index = -1` and NaN bbox/score — so the cache can tell
    "ran and found nothing" apart from "never ran".
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    # Collapse overlapping boxes on the same pole. Class-agnostic — we only
    # ever query one prompt, so the label is the same for every detection.
    boxes_t = results["boxes"]
    scores_t = results["scores"]
    labels_list = list(results["labels"])
    if len(boxes_t) > 0:
        keep = nms(boxes_t, scores_t, nms_iou).tolist()
        boxes_t = boxes_t[keep]
        scores_t = scores_t[keep]
        labels_list = [labels_list[i] for i in keep]

    boxes = boxes_t.tolist()
    scores = scores_t.tolist()
    labels = labels_list

    provenance = {
        "model_id": MODEL_ID,
        "prompt": prompt,
        "threshold": threshold,
        "text_threshold": text_threshold,
        "nms_iou": nms_iou,
    }

    rows: list[dict] = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        rows.append({
            "image_filename": image_path.name,
            "box_index": i,
            "bbox_x1": float(x1),
            "bbox_y1": float(y1),
            "bbox_x2": float(x2),
            "bbox_y2": float(y2),
            "score": float(score),
            "label": str(label),
            **provenance,
        })
    if not rows:
        rows.append({
            "image_filename": image_path.name,
            "box_index": -1,
            "bbox_x1": None,
            "bbox_y1": None,
            "bbox_x2": None,
            "bbox_y2": None,
            "score": None,
            "label": None,
            **provenance,
        })
    return rows
