from uha.data.dataset import make_interleaved_dataset, make_single_dataset
from uha.data.oxe import make_oxe_dataset_kwargs_and_weights
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from uha.pytorch_oxe_dataloader import TorchRLDSIterableDataset
from uha.data.utils.data_utils import NormalizationType
import tensorflow as tf

import torch.nn as nn
import dlimp as dl

tf.config.set_visible_devices([], "GPU")

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
    if not "action_proprio_normalization_type" in cfg:
        action_proprio_normalization_type = NormalizationType("normal")
    else:
        assert cfg.action_proprio_normalization_type == "normal" or cfg.action_proprio_normalization_type == "bounds", "Error in Config, action_proprio_normalization_type should be \"normal\" or \"bounds\""
        action_proprio_normalization_type = NormalizationType(cfg.action_proprio_normalization_type)
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        action_proprio_normalization_type=action_proprio_normalization_type,
        load_camera_views=cfg.load_camera_views,
        dataset_size_limit=cfg.dataset_size_limit if "dataset_size_limit" in cfg else None,
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

def get_single_dataset_tensorflow(cfg: DictConfig, train: bool):
    if not "action_proprio_normalization_type" in cfg:
        action_proprio_normalization_type = NormalizationType("normal")
    else:
        assert cfg.action_proprio_normalization_type == "normal" or cfg.action_proprio_normalization_type == "bounds", "Error in Config, action_proprio_normalization_type should be \"normal\" or \"bounds\""
        action_proprio_normalization_type = NormalizationType(cfg.action_proprio_normalization_type)
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
        cfg.DATA_NAME,
        cfg.DATA_PATH,
        action_proprio_normalization_type=action_proprio_normalization_type,
        load_camera_views=cfg.load_camera_views,
    )

    # create instance of interleaved_dataset_cfg for transforms to work
    interleaved_dataset_cfg = OmegaConf.to_object(cfg.interleaved_dataset_cfg)

    dataset = make_single_dataset(
        dataset_kwargs=dataset_kwargs_list[0],
        train=train,
        traj_transform_kwargs=interleaved_dataset_cfg["traj_transform_kwargs"],
        frame_transform_kwargs=interleaved_dataset_cfg["frame_transform_kwargs"],
        batch_size=1
    )

    return dataset
