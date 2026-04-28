from scripts import build_pole_catalog, download_street_images

def initalize_project():
    """
    Download fresh copies of data to /data directory located 1 level above the current project root
    :return:
    """
    # ----- Download Street Images
    download_street_images.main()

    # ----- Download GIS Pole Layers
    build_pole_catalog.main()

if __name__ == "__main__":
    initalize_project()