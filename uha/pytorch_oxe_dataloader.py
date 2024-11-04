"""
PyTorch dataloader implementation for Open X-Embodiment datasets.
Provides efficient data loading and processing for PyTorch training.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, Union
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
from dlimp.dataset import DLataset

from uha.data.utils.data_utils import hydra_get_object
from uha.data.language_encoders.no_encoder import NoEncoder

logger = logging.getLogger(__name__)

from dataclasses import dataclass
from uha.data.oxe.transforms.base import ImageKeyConfig


@dataclass
class DataloaderConfig:
    """Configuration for PyTorch dataloader."""
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    prefetch_factor: int = 8
    is_single_dataset: bool = False


@dataclass
class TransformConfig:
    """Configuration for transforms."""
    robot_name: str
    action_space: str
    num_arms: int = 1
    image_keys: ImageKeyConfig = ImageKeyConfig()
    depth_keys: Optional[ImageKeyConfig] = None
    gripper_threshold: float = 0.05


class ImageProcessor:
    """Handles image processing operations."""

    @staticmethod
    def move_channel_axis(image: np.ndarray) -> np.ndarray:
        """Move channel axis to PyTorch format (CHW)."""
        return np.moveaxis(image, -1, -3)


class TransformProcessor:
    """Handles data transformation operations."""

    def __init__(self, config: TransformConfig):
        self.config = config
        self.image_processor = ImageProcessor()

    def process_images(self, sample: Dict[str, Any], key: str) -> None:
        """Process images in sample dictionary."""
        if self.config.move_axis:
            for img_key in ["image_primary", "image_secondary", "image_wrist"]:
                if img_key in sample[key]:
                    sample[key][img_key] = self.image_processor.move_channel_axis(
                        sample[key][img_key]
                    )

    def adjust_action_type(self, sample: Dict[str, Any]) -> None:
        """Adjust action data type if specified."""
        if self.config.adjust_type:
            dtype = hydra_get_object(self.config.adjust_type)
            sample["action"] = sample["action"].astype(dtype)

    def process_language(self, sample: Dict[str, Any], 
                        language_encoder: nn.Module) -> None:
        """Process language data in sample."""
        if not self.config.bytes_to_string:
            return

        if sample["task"]["pad_mask_dict"]["language_instruction"]:
            text = sample["task"]["language_instruction"].decode("utf-8")
            sample["task"]["language_instruction"] = language_encoder(text)
        else:
            sample["task"]["language_instruction"] = language_encoder("")

    def process_robot_info(self, sample: Dict[str, Any], 
                          language_encoder: nn.Module) -> None:
        """Process robot information in sample."""
        if "robot_information" not in sample["observation"]:
            return

        if self.config.add_robot_information:
            sample["observation"]["robot_information"] = language_encoder(
                sample["observation"]["robot_information"]
            )
        else:
            del sample["observation"]["robot_information"]

    def remap_keys(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Remap dictionary keys according to configuration."""
        if not self.config.key_remapping:
            if self.config.add_empty_key:
                for key in self.config.add_empty_key:
                    sample[key] = {}
            if "dataset_name" in sample:
                del sample["dataset_name"]
            return sample

        transformed = {}
        if self.config.add_empty_key:
            for key in self.config.add_empty_key:
                transformed[key] = {}

        # Process remapping
        for old_key, value in self.config.key_remapping.items():
            if isinstance(value, dict):
                self._process_nested_remapping(transformed, sample, old_key, value)
            else:
                self._process_single_remapping(transformed, sample, old_key, value)

        return transformed

    def _process_nested_remapping(self, transformed: Dict, sample: Dict, 
                                old_key: str, value: Dict) -> None:
        """Process nested key remapping."""
        for second_old_key, new_value in value.items():
            if isinstance(new_value, list):
                if len(new_value) == 2:
                    transformed.setdefault(new_value[0], {})[new_value[1]] = \
                        sample[old_key][second_old_key]
                elif len(new_value) == 1:
                    transformed[new_value[0]] = sample[old_key][second_old_key]
            else:
                transformed[new_value] = sample[old_key][second_old_key]

    def _process_single_remapping(self, transformed: Dict, sample: Dict,
                                old_key: str, value: Union[str, list]) -> None:
        """Process single key remapping."""
        if isinstance(value, list):
            if len(value) == 2:
                transformed.setdefault(value[0], {})[value[1]] = sample[old_key]
            elif len(value) == 1:
                transformed[value[0]] = sample[old_key]
        else:
            transformed[value] = sample[old_key]


class TorchRLDSIterableDataset(IterableDataset):
    """PyTorch IterableDataset wrapper for RLDS datasets."""

    def __init__(
            self,
            rlds_dataset: DLataset,
            transform_config: Optional[TransformConfig] = None,
            language_encoder: Optional[nn.Module] = None,
            is_single_dataset: bool = False,
            train: bool = True
    ):
        super().__init__()
        self.rlds_dataset = rlds_dataset
        self.transform_config = transform_config or TransformConfig()
        self.language_encoder = language_encoder or NoEncoder()
        self.is_single_dataset = is_single_dataset
        self.is_train = train
        self.current_length = 0
        
        self.transform_processor = TransformProcessor(self.transform_config)

    def __iter__(self):
        """Iterate over dataset with transformations."""
        for sample in self.rlds_dataset.iterator(prefetch=2048):
            if self.is_single_dataset:
                self.current_length = sample["action"].shape[0]
                for i in range(self.current_length):
                    sub_batch = self._limit_size(sample, {}, i)
                    yield self.transform_processor.remap_keys(
                        self._transform_sample(sub_batch)
                    )
            else:
                sample = self._transform_sample(sample)
                yield self.transform_processor.remap_keys(sample)

    def __len__(self) -> int:
        """Get dataset length."""
        if hasattr(self.rlds_dataset, "dataset_len"):
            return self.rlds_dataset.dataset_len

        lengths = np.array([
            stats["num_transitions"]
            for stats in self.rlds_dataset.dataset_statistics
        ])
        
        if hasattr(self.rlds_dataset, "sample_weights"):
            lengths = np.array(self.rlds_dataset.sample_weights) * lengths
            
        total_len = int(lengths.sum())
        return int(0.95 * total_len) if self.is_train else int(0.05 * total_len)

    def _limit_size(self, sample: Any, sub_batch: Dict, index: int) -> Dict:
        """Process sample size limitations."""
        if isinstance(sample, np.ndarray):
            return sample[index]
            
        return {
            key: self._limit_size(value, sub_batch.get(key, {}), index)
            for key, value in sample.items()
        }

    def _transform_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply all transformations to sample."""
        # Process images
        for key in ["observation", "task", "future_obs"]:
            if key in sample:
                self.transform_processor.process_images(sample, key)

        # Adjust action type
        self.transform_processor.adjust_action_type(sample)

        # Process language data
        self.transform_processor.process_language(sample, self.language_encoder)

        # Process robot information
        self.transform_processor.process_robot_info(sample, self.language_encoder)

        return sample


def create_dataloader(
        dataset: DLataset,
        config: DataloaderConfig,
        transform_config: Optional[TransformConfig] = None,
        language_encoder: Optional[nn.Module] = None,
        main_process: bool = False,
        train: bool = True
) -> DataLoader:
    """Create PyTorch DataLoader with configuration."""
    torch_dataset = TorchRLDSIterableDataset(
        rlds_dataset=dataset,
        transform_config=transform_config,
        language_encoder=language_encoder,
        is_single_dataset=config.is_single_dataset,
        train=train
    )

    # Configure workers based on process type
    effective_workers = config.num_workers if main_process else 0

    return DataLoader(
        torch_dataset,
        batch_size=config.batch_size,
        num_workers=effective_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        prefetch_factor=config.prefetch_factor,
        shuffle=None if not config.is_single_dataset else False
    )


# Convenience function for multi-worker setup
def create_multi_worker_dataloader(
        dataset: DLataset,
        config: DataloaderConfig,
        transform_config: Optional[TransformConfig] = None,
        language_encoder: Optional[nn.Module] = None,
        main_process: bool = False,
        train: bool = True
) -> DataLoader:
    """Create multi-worker DataLoader with proper initialization."""
    torch_dataset = TorchRLDSIterableDataset(
        rlds_dataset=dataset,
        transform_config=transform_config,
        language_encoder=language_encoder,
        is_single_dataset=config.is_single_dataset,
        train=train
    )

    def worker_init_fn(worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        worker_info.dataset = torch_dataset

    effective_workers = 2 if main_process else 0

    return DataLoader(
        torch_dataset,
        batch_size=config.batch_size,
        num_workers=effective_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
        prefetch_factor=config.prefetch_factor,
        shuffle=None if not config.is_single_dataset else False,
        worker_init_fn=worker_init_fn
    )