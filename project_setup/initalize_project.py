from config import paths
from scripts import build_pole_catalog, download_street_images

# Every shared directory the pipeline expects under ../data/ (per-route dirs are
# created by the stages that write them). Parents are created along the way.
DATA_DIRS = (
    paths.ROUTES_DIR,
    paths.NEW_ROUTES_DIR,        # inbox: drop annotation XMLs here
    paths.ONBOARDED_ROUTES_DIR,
    paths.STREET_IMAGES_DIR,
    paths.DETECTION_DIR,
)

# RUN THIS SCRIPT BEFORE ANYTHING ELSE

def create_data_dirs():
    """Create any missing data dirs under ../data/, one level above the project root."""
    for path in DATA_DIRS:
        existed = path.is_dir()
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {'exists' if existed else 'created'}: {path}")


def is_initalized() -> bool:
    """Return True if the project has already been initialized (all data dirs exist)."""
    already_initalized = False

    # Check directories have already been created
    if all(path.is_dir() for path in DATA_DIRS):
        already_initalized = True

    return already_initalized


def initalize_project_if_needed(force_reinitalization: bool = False):
    """

    Initial Project Setup. Will not re-run if already run, unless force = true.

    :return:
    """
    print("----- Begin Project Initialization -----")

    if force_reinitalization or not is_initalized():
        create_data_dirs()
    else:
        print("  already initialized, skipping initial setup")

    print("----- Complete Project Initialization -----")


if __name__ == "__main__":
    initalize_project_if_needed()
