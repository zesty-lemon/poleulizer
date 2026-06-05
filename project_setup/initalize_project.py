from pathlib import Path

from scripts import build_pole_catalog, download_street_images

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT.parent / "data"
DATA_SUBDIRS = ("pole_catalog", "route_annotations", "street_images")


def create_data_dirs():
    """Create /data and its top-level subdirs one level above the project root."""
    for name in DATA_SUBDIRS:
        path = DATA_DIR / name
        existed = path.is_dir()
        path.mkdir(parents=True, exist_ok=True)
        print(f"  {'exists' if existed else 'created'}: {path}")


# Todo: Add a method to check for the presence of routes perhaps?
# Todo: Add documentation for where images come from

def initalize_project():
    """
    Download fresh copies of data to /data directory located 1 level above the current project root
    :return:
    """
    print("----- Begin Project Initialization -----")
    create_data_dirs()

    # ----- Download Street Images
    download_street_images.main()

    # ----- Download GIS Pole Layers
    build_pole_catalog.main()
    print("----- Complete Project Initialization -----")

if __name__ == "__main__":
    initalize_project()
