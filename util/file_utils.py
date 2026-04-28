import os
import sys
from pathlib import Path


def delete_ds_store(target_dir):
    """
    Delete any lingering MacOS Preview Files (these break the torchvision loaders)
    These .DS_STORE files are MacOS files used by Spotlight for previews and indexing
    Opening the /data directory will trigger the rebuilding of these files every times
    It is useful to have a utility to delete them if needed
    :param target_dir: Root directory of data (absolute path)
    :return: None
    """
    print("Deleting .DS_Store files (MacOS Preview Files)")
    DATA_DIR = Path(target_dir)

    if not os.path.isdir(DATA_DIR):
        print(f"Error: '{DATA_DIR}' is not a valid directory.")
        sys.exit(1)

    deleted = 0
    for root, dirs, files in os.walk(DATA_DIR):
        for filename in files:
            if filename == ".DS_Store":
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    print(f"Deleted: {filepath}")
                    deleted += 1
                except OSError as e:
                    print(f"Error deleting {filepath}: {e}")

    if deleted == 0:
        print("No .DS_Store files found.")
    else:
        print(f"\nDone. Deleted {deleted} .DS_Store file(s).")

