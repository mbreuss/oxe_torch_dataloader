from uha.data.dataset import make_interleaved_dataset
from uha.data.oxe import make_oxe_dataset_kwargs_and_weights
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from uha.pytorch_oxe_dataloader import TorchRLDSIterableDataset

import torch.nn as nn
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


def make_pytorch_oxe_iterable_dataset(dataset: dl.DLataset, language_encoder: nn.Module = None, train=True, batch_size=512, transform_dict=None, num_workers=0, pin_memory=False, drop_last=False):
    if language_encoder is not None:
        torch_itarable = TorchRLDSIterableDataset(dataset, train, transform_dict, language_encoder=language_encoder)
    else:
        torch_itarable = TorchRLDSIterableDataset(dataset, train, transform_dict)
    
    return DataLoader(
        torch_itarable,
        batch_size=batch_size,
        num_workers=num_workers,  # important to keep this to 0 so PyTorch does not mess with the parallelism
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_octo_dataset_tensorflow(cfg: DictConfig, train: bool):
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        load_camera_views=cfg.load_camera_views,
    )

    # create instance of interleaved_dataset_cfg for transforms to work
    interleaved_dataset_cfg = OmegaConf.to_object(cfg.interleaved_dataset_cfg)

    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=train,
        **interleaved_dataset_cfg,
    )

    return dataset
