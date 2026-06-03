print("STEP 0: script started", flush=True)

import torch
print(f"STEP 1: torch imported ({torch.__version__})", flush=True)

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
print("STEP 2: transformers imported", flush=True)

from PIL import Image, ImageDraw
print("STEP 3: PIL imported", flush=True)

# Force CPU until we have a stable baseline; flip to MPS later if needed.
device = "cpu"
print(f"STEP 4: using device {device}", flush=True)

processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
print("STEP 5: processor loaded", flush=True)

model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base"
).to(device)
print("STEP 6: model loaded", flush=True)

image = Image.open(
    "/Users/giles/Documents/Grad_School/Research/pole_project/data/street_images/000004281227_ROW_20240619.120109.jpg")
print(f"STEP 7: image opened, size={image.size}", flush=True)

inputs = processor(images=image, text="utility pole.", return_tensors="pt").to(device)
print("STEP 8: preprocessed", flush=True)

with torch.no_grad():
    outputs = model(**inputs)
print("STEP 9: inference done", flush=True)

results = processor.post_process_grounded_object_detection(
    outputs,
    threshold=0.25,
    text_threshold=0.25,
    target_sizes=[image.size[::-1]],
)
print("STEP 10: postprocessed", flush=True)
print(results[0]["boxes"])

# Draw boxes + confidence labels on a copy of the input frame and open it in Preview.
annotated = image.copy()
draw = ImageDraw.Draw(annotated)
boxes = results[0]["boxes"].tolist()
scores = results[0]["scores"].tolist()
labels = results[0]["labels"]

for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline="lime", width=8)
    draw.text((x1, max(0, y1 - 40)), f"{label} {score:.2f}", fill="lime")

print(f"STEP 11: drew {len(boxes)} boxes; opening viewer", flush=True)
annotated.show()
