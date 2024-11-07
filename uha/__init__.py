import logging
from typing import Optional, Dict, Any

import torch
import tensorflow as tf
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from uha.data.dataset import make_interleaved_dataset, make_single_dataset
from uha.data.oxe import make_oxe_dataset_kwargs_and_weights
from uha.pytorch_oxe_dataloader import (
    TorchRLDSIterableDataset,
    TransformConfig,
    create_dataloader
)

from uha.data.dataset import SingleDatasetConfig
from uha.data.oxe.oxe_dataset_configs import ImageConfig
from uha.data.utils.data_utils import NormalizationType

# Configure logging
logger = logging.getLogger(__name__)

# Disable TensorFlow GPU usage
tf.config.set_visible_devices([], "GPU")


def get_octo_dataset_tensorflow(cfg: DictConfig, train: bool = True):
    """Get OXE dataset with TensorFlow backend.
    
    Args:
        cfg: Configuration object containing dataset specs
        train: Whether this is for training or validation
        
    Returns:
        Configured dataset ready for training/validation
    """
    try:
        # Determine normalization type
        norm_type = getattr(cfg, "action_proprio_normalization_type", "normal")
        if norm_type not in {"normal", "bounds"}:
            raise ValueError(f"Invalid normalization type: {norm_type}")
            
        action_proprio_normalization_type = NormalizationType(norm_type)

        # Get dataset kwargs and weights
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
            cfg.DATA_NAME,
            cfg.DATA_PATH,
            action_proprio_normalization_type=action_proprio_normalization_type,
            load_camera_views=cfg.load_camera_views,
            dataset_size_limit=getattr(cfg, "dataset_size_limit", None),
        )

        # Adjust shuffle buffer size for validation
        if not train:
            cfg.interleaved_dataset_cfg.shuffle_buffer_size //= 100

        # Create dataset
        dataset = make_interleaved_dataset(
            dataset_kwargs_list,
            sample_weights,
            # train=train,
            **OmegaConf.to_object(cfg.interleaved_dataset_cfg)
        )

        return dataset

    except Exception as e:
        logger.error(f"Error creating OXE dataset: {str(e)}")
        raise


def make_pytorch_oxe_iterable_dataset(
        dataset,
        train: bool = True,
        batch_size: int = 512,
        transform_config: Optional[TransformConfig] = None,
        language_encoder: Optional[torch.nn.Module] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        is_single_dataset: bool = False,
        main_process: bool = False,
        prefetch_factor: int = 8
) -> DataLoader:
    """Create PyTorch DataLoader from RLDS dataset with improved configuration"""
    
    torch_iterable = TorchRLDSIterableDataset(
        rlds_dataset=dataset,
        train=train,
        transform_config=transform_config,
        language_encoder=language_encoder,
        is_single_dataset=is_single_dataset
    )

    # Configure num_workers based on process type
    effective_workers = 1 if main_process else 0
    
    return create_dataloader(
        dataset=torch_iterable,
        batch_size=batch_size,
        num_workers=effective_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        shuffle=None if not is_single_dataset else False
    )


def get_oxe_dataset_tensorflow(cfg: DictConfig, train: bool):
    """Get OXE dataset with TensorFlow backend"""
    try:
        # Determine normalization type
        norm_type = getattr(cfg, "action_proprio_normalization_type", "normal")
        if norm_type not in {"normal", "bounds"}:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        
        action_proprio_normalization_type = NormalizationType(norm_type)

        # Get dataset kwargs and weights
        dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
            cfg.DATA_NAME,
            cfg.DATA_PATH,
            action_proprio_normalization_type=action_proprio_normalization_type,
            load_camera_views=cfg.load_camera_views,
            dataset_size_limit=getattr(cfg, "dataset_size_limit", None),
        )

        # Adjust shuffle buffer size for validation
        if not train:
            cfg.interleaved_dataset_cfg.shuffle_buffer_size //= 100

        # Create dataset
        dataset = make_interleaved_dataset(
            dataset_kwargs_list,
            sample_weights,
            train=train,
            **cfg.interleaved_dataset_cfg
        )

        return dataset

    except Exception as e:
        logger.error(f"Error creating OXE dataset: {str(e)}")
        raise


def get_single_dataset_tensorflow(cfg: DictConfig, train: bool = True):
    """Get single OXE dataset with TensorFlow backend."""
    try:
        # Determine normalization type
        norm_type = getattr(cfg, "action_proprio_normalization_type", "normal")
        if norm_type not in {"normal", "bounds"}:
            raise ValueError(f"Invalid normalization type: {norm_type}")
            
        action_proprio_normalization_type = NormalizationType(norm_type)

        # Get dataset kwargs
        dataset_kwargs_list, _ = make_oxe_dataset_kwargs_and_weights(
            cfg.DATA_NAME,
            cfg.DATA_PATH,
            action_proprio_normalization_type=action_proprio_normalization_type,
            load_camera_views=cfg.load_camera_views,
        )

        logger.info(f"Constructing single {'train' if train else 'val'} dataset: {dataset_kwargs_list[0]['name']}")
        
        # Prepare dataset kwargs
        dataset_kwargs = dataset_kwargs_list[0].copy()
        dataset_kwargs["train"] = train
        
        # Create config
        config = SingleDatasetConfig(
            train=train,
            traj_transform_kwargs=cfg.interleaved_dataset_cfg["traj_transform_kwargs"],
            frame_transform_kwargs=cfg.interleaved_dataset_cfg["frame_transform_kwargs"]
        )

        return make_single_dataset(
            dataset_kwargs=dataset_kwargs,
            config=config
        )

    except Exception as e:
        logger.error(f"Error creating single dataset: {str(e)}")
        raise