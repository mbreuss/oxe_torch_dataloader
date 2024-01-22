from Uha.data.dataset import make_interleaved_dataset
from Uha.data.oxe import make_oxe_dataset_kwargs_and_weights
from torch.utils.data import DataLoader
from Uha.pytorch_oxe_dataloader import TorchRLDSIterableDataset

import tqdm
import tensorflow as tf
import tensorflow_datasets as tfds

DATA_NAME = "oxe_magic_soup"
DATA_PATH = "gs://gresearch/robotics"  # "gs://rail-orca-central2/resize_256_256"
# DOWNLOAD_DIR = '~/tensorflow_datasets'

tf.config.set_visible_devices([], "GPU")


def download_oxe_data(download_dir):
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        # DATA_PATH + "/" + DATA_NAME,
        DATA_NAME,
        download_dir,
        load_camera_views=("primary", "wrist"),
    )

    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    # go through datasets once to get sizes
    for dataset_kwargs in dataset_kwargs_list:
        if (dataset_kwargs["name"] == "fractal20220817_data" or dataset_kwargs["name"] == "kuka"
                or dataset_kwargs["name"] == "bridge_dataset"):
            pass
        else:
            _ = tfds.load(name=dataset_kwargs["name"], data_dir=dataset_kwargs["data_dir"], download=True)


def download_oxe_data_builder(download_dir):
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        # DATA_PATH + "/" + DATA_NAME,
        DATA_NAME,
        download_dir,
        load_camera_views=("primary", "wrist"),
    )

    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    # go through datasets once to get sizes
    for dataset_kwargs in dataset_kwargs_list:
        if (dataset_kwargs["name"] == "fractal20220817_data" or dataset_kwargs["name"] == "kuka"
                or dataset_kwargs["name"] == "bridge_dataset"):
            pass
        else:
            download_and_prepare_kwargs = {'download_config': tfds.core.download.DownloadConfig(try_download_gcs=False)}
            builder = tfds.builder(name=dataset_kwargs["name"], data_dir=dataset_kwargs["data_dir"])
            # download_and_prepare_kwargs = {'download_config': tfds.core.download.DownloadConfig(try_download_gcs=False)}
            builder.download_and_prepare()


def make_pytorch_oxe_iterable_dataset(dataset=None, batch_size=512):
    if dataset is None:
        dataset = get_octo_dataset_tensorflow()

    pytorch_dataset = TorchRLDSIterableDataset(dataset)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

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
