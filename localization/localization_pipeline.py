from dataset_util import annotations_util
from util import localization_utils


def localize_poles():
    # Load Pole Annotations
    annotations_df = annotations_util.load_annotations()

    for row in annotations_df.itertuples():
        print(row.Index, row.filename, row.source_lat, row.source_lon) # ignore warnings

    # for offset in range(1, MAX_WALK_BACK):
    #     candidate = df.iloc[labeled_idx - offset]
    #     if candidate.batch != df.iloc[labeled_idx].batch:
    #         break  # crossed a batch boundary
    #     if not in_fov(candidate, pole):
    #         break
if __name__ == "__main__":
    localize_poles()


