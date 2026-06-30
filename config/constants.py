# ----- Rigid-feature label codes (from VAIL annotations) -----
TREE = 1
UTILITY_POLE = 2
LEDGE = 3
EMBANKMENT = 4
ROCK = 5
OTHER = 6
NOT_APPLICABLE = 98
UNKNOWN_OBSCURED = 99

# ----- Street-image pixel dimensions -----
# Verified uniform across the dataset (every sampled JPG is 3840x2160). Only the
# width matters for the horizontal angle->pixel projection; height is unused by
# matching. EXIF carries no camera intrinsics, so these are measured, not read.
IMAGE_WIDTH_PX = 3840
IMAGE_HEIGHT_PX = 2160

