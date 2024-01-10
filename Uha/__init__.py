from Uha.data.dataset import make_interleaved_dataset
from Uha.data.oxe import make_oxe_dataset_kwargs_and_weights
from torch.utils.data import DataLoader
from Uha.pytorch_oxe_dataloader import TorchRLDSIterableDataset, TorchRLDSTensorDataset

import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

DATA_NAME = "oxe_magic_soup"
DATA_PATH = "gs://gresearch/robotics"  # "gs://rail-orca-central2/resize_256_256"
DOWNLOAD_DIR = '~/tensorflow_datasets'

tf.config.set_visible_devices([], "GPU")


def download_oxe_data():
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        DATA_NAME,
        DATA_PATH,
        # DOWNLOAD_DIR,
        load_camera_views=("primary", "wrist"),
    )

    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = []
    for dataset_kwargs in dataset_kwargs_list:
        REQUIRED_KEYS = {"observation", "action"}
        # if dataset_kwargs.__getitem__("language_key") is not None:
        #     REQUIRED_KEYS.add(dataset_kwargs['language_key'])
        #
        # _, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
        #
        # builder = tfds.builder(name, data_dir=data_dir)
        # tfds.load()
        #
        # dataset_sizes.append(dataset_statistics["num_transitions"])
        # all_dataset_statistics.append(dataset_statistics)
        tfds.load(**dataset_kwargs)


def make_pytorch_oxe_tensor_dataset(dataset=None):
    if dataset is None:
        dataset = get_octo_dataset_tensorflow()

    pytorch_dataset = TorchRLDSTensorDataset(dataset)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=16,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

    for i, sample in tqdm.tqdm(enumerate(dataloader)):
        print(sample)
        if i == 5000:
            break

    return dataloader


def make_pytorch_oxe_iterable_dataset(dataset=None):
    if dataset is None:
        dataset = get_octo_dataset_tensorflow()

    pytorch_dataset = TorchRLDSIterableDataset(dataset)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=16,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

    for i, sample in tqdm.tqdm(enumerate(dataloader)):
        print(sample)
        if i == 5000:
            break

    return dataloader


def get_octo_dataset_tensorflow():
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        DATA_NAME,
        DATA_PATH,
        # DOWNLOAD_DIR,
        load_camera_views=("primary", "wrist"),
    )

    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=True,
        shuffle_buffer_size=1000,
        # change to 500k for training, large shuffle buffers are important, but adjust to your RAM
        batch_size=None,  # batching will be handles in PyTorch Dataloader object
        balance_weights=True,
        traj_transform_kwargs=dict(
            goal_relabeling_strategy="uniform",
            window_size=2,
            future_action_window_size=3,
            subsample_length=100,
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs={
                "primary": dict(
                    random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
                "wrist": dict(
                    random_brightness=[0.1],
                    random_contrast=[0.9, 1.1],
                    random_saturation=[0.9, 1.1],
                    random_hue=[0.05],
                    augment_order=[
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                ),
            },
            resize_size=dict(
                primary=(256, 256),
                wrist=(128, 128),
            ),
            num_parallel_calls=200,
        ),
        traj_transform_threads=48,
        traj_read_threads=48,
    )

    return dataset


make_pytorch_oxe_iterable_dataset()
