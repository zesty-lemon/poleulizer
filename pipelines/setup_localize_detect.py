from project_setup.initalize_project import initalize_project_if_needed



def initial_setup():
    """
    Initial project setup if desired
    :return:
    """
    initalize_project_if_needed()


def run_setup_localize_detect(run_initial_setup: bool = False,
                              download_street_images: bool = False,
                              build_pole_catalog: bool = False,
                              localize_poles: bool = True,
                              detect_poles: bool = True
                              ):
    """
    Run initial setup (if desired), download street images (if desired), build pole catalog,
    localize the street poles, and detect the presence of poles

    :param run_initial_setup: Creates ../data/ directories, downloads images, builds image catalog
    :param download_street_images: Mirrors every JPG referenced by XMLs in new_routes/, then archives the XML to onboarded_routes/
    :param build_pole_catalog: Geohashes every labeled pole into a deduped catalog keyed by geohash:<9 chars>
    :param localize_poles: For each labeled pole, walks back through frames and records every frame where the pole is in the camera FOV
    :param detect_poles: Runs Grounding-DINO on each observation image and caches the bounding boxes. Idempotent (skips cached images)
    :return: None
    """

    if run_initial_setup:
        initial_setup()


"""
Maybe we should have intial setup. Then onboard new routes. 

Initial Setup: Just create the directories
Onboard new routes: Download street, build pole, localize pole, detect pole. 

    create_data_dirs()

    # ----- Download Street Images
    download_street_images.main()

    # ----- Build Pole Catalog (geohash labeled poles from onboarded XMLs)
    build_pole_catalog.main()
    

"""