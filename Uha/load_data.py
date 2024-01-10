# minimum working example to load a single OXE dataset
from Uha.data.oxe import make_oxe_dataset_kwargs
from Uha.data.dataset import make_single_dataset


def main():
    dataset_kwargs = make_oxe_dataset_kwargs(
        # see octo/data/oxe/oxe_dataset_configs.py for available datasets
        # (this is a very small one for faster loading)
        "austin_buds_dataset_converted_externally_to_rlds",
        # can be local or on cloud storage (anything supported by TFDS)
        # "/path/to/base/oxe/directory",
        "gs://gresearch/robotics",
    )
    dataset = make_single_dataset(dataset_kwargs, train=True)  # load the train split
    iterator = dataset.iterator()
    traj = next(iterator)
    print("Top-level keys: ", traj.keys())
    print("Observation keys: ", traj["observation"].keys())
    print("Task keys: ", traj["task"].keys())
    images = traj["observation"]["image_primary"]
    # should be: (traj_len, window_size, height, width, channels)
    # (window_size defaults to 1)
    print(images.shape)

    # "/media/toradus/Volume/datasets",


if __name__ == "__main__":
    main()
