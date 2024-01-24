from uha.data.dataset import make_interleaved_dataset
from uha.data.oxe import make_oxe_dataset_kwargs_and_weights
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from uha.pytorch_oxe_dataloader import TorchRLDSIterableDataset

import tensorflow as tf
import tensorflow_datasets as tfds
import dlimp as dl

tf.config.set_visible_devices([], "GPU")


def download_oxe_data(cfg: DictConfig):
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.download_dir,
        load_camera_views=cfg.load_camera_views,
    )

    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    # go through datasets once to get sizes
    for dataset_kwargs in dataset_kwargs_list:
        _ = tfds.load(name=dataset_kwargs["name"], data_dir=dataset_kwargs["data_dir"], download=True)


def make_pytorch_oxe_iterable_dataset(dataset: dl.DLataset, batch_size=512):
    pytorch_dataset = TorchRLDSIterableDataset(dataset)
    dataloader = DataLoader(
        pytorch_dataset,
        batch_size=batch_size,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

    return dataloader


def get_octo_dataset_tensorflow(cfg: DictConfig, train: bool):
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        load_camera_views=cfg.load_camera_views,
    )

    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        **cfg.interleaved_dataset_cfg,
    )

    return dataset


def get_octo_dataset_tensorflow_test(cfg: DictConfig, train: bool):
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        load_camera_views=cfg.load_camera_views,
    )

    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
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
                ),**cfg.interleaved_dataset_cfg,
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


def get_octo_dataset_tensorflow_old(train: bool):
    DATA_NAME = "oxe_magic_soup"
    DATA_PATH = "gs://gresearch/robotics"
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        DATA_NAME,
        DATA_PATH,
        load_camera_views=("primary", "wrist"),
    )

    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
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
