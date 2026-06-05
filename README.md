# Poleulizer
A Masters Project by Giles Lemmon

A pipeline for **localizing utility poles in Vermont dashcam footage**.

Given VAIL-produced annotation batches (XML) and the POV street images they
reference, the project finds where each labeled pole appears in the image
sequence and assigns each pole a stable ID — so that, for any pole, you can
pull every frame and bounding box that shows it.


---

## The big idea

VAIL labels each pole **once**, at the frame where the vehicle is right next to
it — which is the one frame where the pole is *off to the side and out of the
forward cameras' view*. To actually see the pole, we walk **backwards** through
earlier frames (while the vehicle was approaching), use the source GPS + heading
to figure out which frames had the pole in the camera's field of view, then run
an object detector to draw bounding boxes on it.

```
VAIL XML ─► download images ─► build pole catalog
                │
                └─► localization (walk back, find frames pole is visible)
                            │
                            └─► detection (bounding boxes on those frames)
```

Full problem framing: [`documentation/localization_problem.md`](documentation/localization_problem.md).

---

## The important jobs (where the work happens)

Each of these is a runnable entry point. Run them in this order.

| Step | Run this | What it does | Output |
| --- | --- | --- | --- |
| **0. Setup** | `python3 project_setup/initalize_project.py` | Creates the `../data/` dirs, downloads images, builds the catalog (steps 1+2 in one shot). | `../data/` tree |
| **1. Download images** | `python3 scripts/download_street_images.py` | Mirrors every JPG referenced by XMLs in `new_routes/`, then archives the XML to `onboarded_routes/`. Idempotent. | `../data/street_images/*.jpg` |
| **2. Build catalog** | `python3 scripts/build_pole_catalog.py` | Geohashes every labeled pole into a deduped catalog keyed by `geohash:<9 chars>`. | `../data/pole_catalog/poles.parquet` |
| **3. Localize** | `python3 localization/localization_pipeline.py` | For each labeled pole, walks back through frames and records every frame where the pole is in the camera FOV. | `../data/localization/observations.parquet` |
| **4. Detect** | `python3 detection/run_detection.py` | Runs Grounding-DINO on each observation image and caches the bounding boxes. Idempotent (skips cached images). | `../data/detection/bboxes.parquet` |

---

## Project layout

```
poleulizer/
├── project_setup/
│   └── initalize_project.py      # JOB: one-shot setup (mkdir data + download + catalog)
│
├── scripts/                       # JOBS: data ingest
│   ├── download_street_images.py # download POV JPGs, archive onboarded XMLs
│   └── build_pole_catalog.py     # geohash labeled poles into a catalog
│
├── localization/
│   └── localization_pipeline.py  # JOB: which frames show each pole (FOV walk-back)
│
├── detection/
│   ├── run_detection.py          # JOB: cache bounding boxes for observation images
│   ├── detection_util.py         #   Grounding-DINO wrapper (load + detect)
│   └── sandbox.py                #   scratch / experiments
│
├── dataset_util/
│   └── annotations_util.py       # XML → pandas DataFrame (load_annotations)
│
├── util/                          # shared helpers
│   ├── localization_utils.py     #   bearing/haversine/in_fov + EXIF GPS reads
│   ├── geohash.py                #   geohash-9 encoder (no deps)
│   └── file_utils.py             #   .DS_Store cleanup
│
├── config/
│   └── constants.py              # VAIL rigid-type codes (UTILITY_POLE=2, ...)
│
├── documentation/                 # design docs — START HERE
│   ├── project_overview.md       #   the canonical overview
│   ├── localization_problem.md   #   the formal problem & geometry
│   ├── detection_design.md
│   ├── pole_catalog.md
│   ├── dataset_documentation.md  #   the VAIL XML schema
│   └── download_street_images_design.md
│
├── info_and_research/             # project plan, meeting notes, methodology
└── temp_junk/                     # parked code (state-utility GIS matching), not in use
```

> **Where the data lives:** all generated data sits *outside* the repo, one level
> up, in `../data/` — so the repo stays code-only. The pipeline reads and writes
> there. The XML annotations are the single source of truth; everything else
> (images on disk, catalog, observations, bboxes) is a regenerable artifact.

---

## Loading the results

```python
import pandas as pd
from dataset_util.annotations_util import load_annotations, load_pole_annotations

annotations = load_annotations()                                  # every annotation row
poles       = load_pole_annotations()                             # just utility-pole rows
catalog     = pd.read_parquet("../data/pole_catalog/poles.parquet")
obs         = pd.read_parquet("../data/localization/observations.parquet")
bboxes      = pd.read_parquet("../data/detection/bboxes.parquet")

# Every detected box joined to its pole/frame observation:
merged = obs.merge(bboxes, on="image_filename", how="left")
```

---

## Status & caveats

- **Working today:** image ingest, geohash catalog, FOV-based localization
  (center camera only for now), and zero-shot Grounding-DINO detection.
- **Placeholder geometry:** camera FOV/yaw values in `localization_pipeline.py`
  are estimates. Accurate camera intrinsics + extrinsics from VAIL are the top
  open blocker — see the "Open questions" in `documentation/localization_problem.md`.
- **Deferred:** mapping geohash IDs to canonical state-utility IDs (the GIS code
  in `temp_junk/`) is out of scope for now.


## AI Disclosure
Generative AI was heavily used to generate documentation after the relevant code had been completed. Documentation
was reviewed after generation. The primary purpose of extensive documentation is to assist in the transfer
of this work to a future student to build off of. LLMS were also used as a research tool on tasks such as understanding
APIs that do not have associated documentation, and crawling state websites to source desired information. 