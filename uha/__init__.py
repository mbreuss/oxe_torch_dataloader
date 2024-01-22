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
