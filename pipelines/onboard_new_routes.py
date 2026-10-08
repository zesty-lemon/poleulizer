from config import paths
from config.paths import RoutePaths
from scripts.build_pole_catalog import run_build_pole_catalog_pipeline
from localization.localization_pipeline import run_localization_pipeline
from detection.run_detection import run_detection_and_save
from project_setup.initalize_project import initalize_project_if_needed
from matching.matching_pipeline import perform_matching
from matching.export_poles import export_complete_and_isolated
from matching.evaluate import run_evaluation_pipeline
from scripts.download_street_images import run_download_street_images_pipeline


def resolve_route(batch: str | None = None) -> RoutePaths:
    """
    Decide which route this run is about and return its RoutePaths.

    One route per run. With `batch` given, it must name an XML either waiting in
    new_routes/ (about to be onboarded) or already archived in onboarded_routes/
    (re-running later stages). Without it, new_routes/ must hold exactly one XML
    and its stem becomes the batch name.

    :param batch: batch name (= XML stem), or None to take the single XML in new_routes/
    :return: RoutePaths for the resolved batch
    """
    inbox = sorted(paths.NEW_ROUTES_DIR.glob("*.xml"))
    archived = sorted(paths.ONBOARDED_ROUTES_DIR.glob("*.xml"))

    if batch is None:
        if len(inbox) != 1:
            found = [p.name for p in inbox] or ["(none)"]
            known = [p.stem for p in archived] or ["(none)"]
            raise RuntimeError(
                f"expected exactly one XML in {paths.NEW_ROUTES_DIR}, found {len(inbox)}: {found}\n"
                f"  pass batch=<name> explicitly to re-run an onboarded route; known routes: {known}")
        batch = inbox[0].stem
    elif not any(p.stem == batch for p in inbox + archived):
        known = [p.stem for p in inbox + archived] or ["(none)"]
        raise RuntimeError(f"no XML named {batch!r} in new_routes/ or onboarded_routes/; known routes: {known}")

    route = paths.route_paths(batch)
    print(f"route: {batch}\n  -> {route.root}")
    return route


def onboard_new_route_and_run_pipeline(batch: str | None = None,
                                       force_reinitalization: bool = False,
                                       download_street_images: bool = True,
                                       build_pole_catalog: bool = True,
                                       localize_poles: bool = True,
                                       detect_poles: bool = True,
                                       perform_pole_matching: bool = True,
                                       export_complete_poles: bool = True,
                                       export_isolated_poles: bool = True,
                                       evaluate_matches: bool = True
                                       ):
    """
    Onboards a new route and runs the full pipeline end to end.

    Main entry point into this project

    Run initial setup (if desired), download street images, build pole catalog,
    localize the street poles, detect the presence of poles, match poles to
    bounding boxes, export the pole-centric deliverable (whole catalog, plus
    the isolated-pole subset), and score the matches if the route has been annotated

    :param batch: Route to process (= XML stem). None -> the single XML waiting in new_routes/.
        See resolve_route for the rules.
    :param force_reinitalization: Forces ../data/ directory creation even if already initialized
    :param download_street_images: Mirrors every JPG referenced by this route's XML in new_routes/, then archives the XML to onboarded_routes/
    :param build_pole_catalog: Geohashes every labeled pole into a deduped catalog keyed by geohash:<9 chars>
    :param localize_poles: For each labeled pole, walks back through frames and records every frame where the pole is in the camera FOV
    :param detect_poles: Runs Grounding-DINO on each observation image and caches the bounding boxes. Idempotent (skips cached images).
        The bbox cache is shared across routes and never cleared here; use run_detection_and_save(clear_cache=True)
        directly when the detector config changes (note: that renumbers box_index and voids any ground truth)
    :param perform_pole_matching: Match poles to bounding boxes
    :param export_complete_poles: Write the pole-centric CSV + per-pole image folders to <route>/pole_export/:
        complete_route/ for the whole catalog, isolated_poles/ for the subset of pole-frames where the
        pole is alone in its frame (computed from the localization observations, no annotation needed)
    :param export_isolated_poles: Include a sub directory for only poles alone in their frames
    :param evaluate_matches: Write the annotation template to <route>/evaluation/; if a filled-in
        ground_truth.csv is there, score the matches and save reports. Skips (no error) otherwise,
        so the first run of a route writes the template and the second run scores it.
    :return: None
    """

    # ----- Initial Project Setup (only runs if needed or forced)
    initalize_project_if_needed(force_reinitalization=force_reinitalization)

    # ----- Resolve the one route this run is about (before download archives its XML)
    route = resolve_route(batch)

    # ----- Download Street Images from VAIL Server
    if download_street_images:
        run_download_street_images_pipeline(route)

    # ----- Build Pole Catalog (geohash labeled poles from this route's XML)
    if build_pole_catalog:
        run_build_pole_catalog_pipeline(route)

    # ----- Localize Poles within visible field of each image
    if localize_poles:
        run_localization_pipeline(route)

    # ----- Put bounding boxes around each pole visible in each frame
    if detect_poles:
        run_detection_and_save(route, clear_cache=False)

    # ----- Perform Matching
    if perform_pole_matching:
        perform_matching(route)

    # ----- Export Deliverable (complete route + isolated subset)
    if export_complete_poles or export_isolated_poles:
        export_complete_and_isolated(route,
                                     include_complete_route=export_complete_poles,
                                     include_isolated_poles=export_isolated_poles)

    # ----- Evaluate (only if this route has a hand-filled ground truth)
    if evaluate_matches:
        run_evaluation_pipeline(route)


if __name__ == "__main__":
    onboard_new_route_and_run_pipeline()