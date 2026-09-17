from config import paths
from scripts import build_pole_catalog, download_street_images

DATA_DIR = paths.DATA_DIR
DATA_SUBDIRS = ("routes", "route_annotations", "street_images")


def create_data_dirs():
    """Create /data and its top-level subdirs one level above the project root."""
    for name in DATA_SUBDIRS:
        path = DATA_DIR / name
        existed = path.is_dir()
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {'exists' if existed else 'created'}: {path}")


def is_initalized() -> bool:
    """Return True if the project has already been initialized (all data dirs exist)."""
    already_initalized = False

    # Check directories have already been created
    if all((DATA_DIR / name).is_dir() for name in DATA_SUBDIRS):
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
