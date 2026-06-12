from scripts.build_pole_catalog import main as build_pole_catalog_main
from scripts.download_street_images import main as download_street_images_main
from localization.localization_pipeline import main as localize_poles_main
from detection.run_detection import main as run_detection_main
from project_setup.initalize_project import initalize_project_if_needed


def onboard_new_route_and_run_localize_detect(force_reinitalization: bool = False,
                                              download_street_images: bool = False,
                                              build_pole_catalog: bool = False,
                                              localize_poles: bool = True,
                                              detect_poles: bool = True
                                              ):
    """
    Onboards a new route and runs localization and detection pipelines.

    Run initial setup (if desired), download street images (if desired), build pole catalog,
    localize the street poles, and detect the presence of poles

    :param force_reinitalization: Forces ../data/ directory creation even if already initialized
    :param download_street_images: Mirrors every JPG referenced by XMLs in new_routes/, then archives the XML to onboarded_routes/
    :param build_pole_catalog: Geohashes every labeled pole into a deduped catalog keyed by geohash:<9 chars>
    :param localize_poles: For each labeled pole, walks back through frames and records every frame where the pole is in the camera FOV
    :param detect_poles: Runs Grounding-DINO on each observation image and caches the bounding boxes. Idempotent (skips cached images)
    :return: None
    """

    # ----- Initial Project Setup (only runs if needed or forced)
    initalize_project_if_needed(force_reinitalization=force_reinitalization)

    # ----- Download Street Images from VAIL Server
    if download_street_images:
        download_street_images_main()

    # ----- Build Pole Catalog (geohash labeled poles from onboarded XMLs)
    if build_pole_catalog:
        build_pole_catalog_main()

    # ----- Localize Poles within visible field of each image
    if localize_poles:
        localize_poles_main()

    # ----- Put bounding boxes around each pole visible in each frame
    if detect_poles:
        run_detection_main()

if __name__ == "__main__":
    onboard_new_route_and_run_localize_detect()