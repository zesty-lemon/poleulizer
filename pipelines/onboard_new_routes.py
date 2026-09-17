from scripts.build_pole_catalog import main as build_pole_catalog_main
from localization.localization_pipeline import run_localization_pipeline
from detection.run_detection import main as run_detection_main
from project_setup.initalize_project import initalize_project_if_needed
from matching.matching_pipeline import perform_matching
from matching.export_poles import export_complete_and_isolated
from scripts.download_street_images import run_download_street_images_pipeline


def onboard_new_route_and_run_pipeline(force_reinitalization: bool = False,
                                       download_street_images: bool = True,
                                       build_pole_catalog: bool = True,
                                       localize_poles: bool = True,
                                       detect_poles: bool = True,
                                       perform_pole_matching: bool = True,
                                       export_complete_poles: bool = True,
                                       export_isolated_poles: bool = True
                                       ):
    """
    Onboards a new route and runs the full pipeline end to end.

    Main entry point into this project

    Run initial setup (if desired), download street images, build pole catalog,
    localize the street poles, detect the presence of poles, match poles to
    bounding boxes, and export the pole-centric deliverable (whole catalog, plus
    the isolated-pole subset)

    :param force_reinitalization: Forces ../data/ directory creation even if already initialized
    :param download_street_images: Mirrors every JPG referenced by XMLs in new_routes/, then archives the XML to onboarded_routes/
    :param build_pole_catalog: Geohashes every labeled pole into a deduped catalog keyed by geohash:<9 chars>
    :param localize_poles: For each labeled pole, walks back through frames and records every frame where the pole is in the camera FOV
    :param detect_poles: Runs Grounding-DINO on each observation image and caches the bounding boxes. Idempotent (skips cached images)
    :param perform_pole_matching: Match poles to bounding boxes
    :param export_complete_poles: Write the pole-centric CSV + per-pole image folders to matching/pole_export:
        complete_route/ for the whole catalog, isolated_poles/ for the subset of pole-frames where the
        pole is alone in its frame (computed from the localization observations, no annotation needed)
    :param export_isolated_poles: Include a sub directory for only poles alone in their frames
    :return: None
    """

    # ----- Initial Project Setup (only runs if needed or forced)
    initalize_project_if_needed(force_reinitalization=force_reinitalization)

    # ----- Download Street Images from VAIL Server
    if download_street_images:
        run_download_street_images_pipeline()

    # ----- Build Pole Catalog (geohash labeled poles from onboarded XMLs)
    if build_pole_catalog:
        build_pole_catalog_main()

    # ----- Localize Poles within visible field of each image
    if localize_poles:
        run_localization_pipeline()

    # ----- Put bounding boxes around each pole visible in each frame
    if detect_poles:
        run_detection_main()

    # ----- Perform Matching
    if perform_pole_matching:
        perform_matching()

    # ----- Export Deliverable (complete route + isolated subset)
    if export_complete_poles or export_isolated_poles:
        export_complete_and_isolated(include_complete_route = export_complete_poles,
                                     include_isolated_poles=export_isolated_poles)


if __name__ == "__main__":
    onboard_new_route_and_run_pipeline()