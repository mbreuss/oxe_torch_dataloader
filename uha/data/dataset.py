"""
Dataset creation and management for Open X-Embodiment.
Handles dataset construction, transformation, and interleaving.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import logging
from pathlib import Path
import json

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from uha.data import obs_transforms, traj_transforms
from uha.data.utils import goal_relabeling, task_augmentation
from uha.data.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    NormalizationType,
    normalize_action_and_proprio,
    pprint_data_mixture,
    sample_match_keys_uniform,
    tree_map
)
from uha.data.utils.spec import ModuleSpec

logger = logging.getLogger(__name__)


def convert_image_config_to_dict(image_config: Any) -> Dict[str, Optional[str]]:
    """Convert ImageConfig to dictionary format."""
    if not hasattr(image_config, 'primary'):
        return image_config  # Already a dict or other format
        
    return {
        k: v for k, v in {
            "primary": image_config.primary,
            "secondary": image_config.secondary,
            "wrist": image_config.wrist
        }.items() if v is not None
    }


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""
    name: str
    data_dir: Union[str, Path]
    train: bool = True
    standardize_fn: Optional[ModuleSpec] = None
    shuffle: bool = True
    image_obs_keys: Mapping[str, Optional[str]] = field(default_factory=dict)
    depth_obs_keys: Mapping[str, Optional[str]] = field(default_factory=dict)
    proprio_obs_key: Optional[str] = None
    language_key: Optional[str] = None
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL
    dataset_statistics: Optional[Union[Dict, str]] = None
    force_recompute_dataset_statistics: bool = False
    action_normalization_mask: Optional[Sequence[bool]] = None
    filter_functions: Sequence[ModuleSpec] = field(default_factory=tuple)
    skip_norm: bool = False
    ignore_errors: bool = False
    num_parallel_reads: int = tf.data.AUTOTUNE
    num_parallel_calls: int = tf.data.AUTOTUNE
    dataset_size_limit: Optional[int] = None

    def __post_init__(self):
        """Validate and process configuration."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory transforms."""
    train: bool
    goal_relabeling_strategy: Optional[str] = None
    goal_relabeling_kwargs: Dict[str, Any] = field(default_factory=dict)
    window_size: int = 1
    action_horizon: int = 1
    subsample_length: Optional[int] = None
    skip_unlabeled: bool = False
    max_action: Optional[float] = None
    max_proprio: Optional[float] = None
    task_augment_strategy: Optional[str] = None
    task_augment_kwargs: Dict[str, Any] = field(default_factory=dict)
    max_action_dim: Optional[int] = None
    max_proprio_dim: Optional[int] = None
    post_chunk_transforms: Sequence[ModuleSpec] = field(default_factory=tuple)
    num_parallel_calls: int = tf.data.AUTOTUNE


@dataclass
class FrameConfig:
    """Configuration for frame transforms."""
    train: bool
    image_augment_kwargs: Union[Dict[str, Any], Mapping[str, Dict[str, Any]]] = \
        field(default_factory=dict)
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = \
        field(default_factory=dict)
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = \
        field(default_factory=dict)
    image_dropout_prob: float = 0.0
    image_dropout_keep_key: Optional[str] = None
    num_parallel_calls: int = tf.data.AUTOTUNE


class DatasetBuilder:
    """Handles dataset creation and processing."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate dataset configuration."""
        required_keys = {"observation", "action"}
        
        # Ensure image_obs_keys is a dictionary
        if hasattr(self.config.image_obs_keys, 'primary'):
            self.config.image_obs_keys = convert_image_config_to_dict(self.config.image_obs_keys)
            
        # Validate keys
        if not isinstance(self.config.image_obs_keys, dict):
            raise ValueError("image_obs_keys must be a dictionary or ImageConfig")
            
        if not self.config.image_obs_keys:
            raise ValueError("image_obs_keys cannot be empty")

    def build(self) -> Tuple[dl.DLataset, Dict[str, Any]]:
        """Build dataset according to configuration."""
        try:
            builder = tfds.builder(
                self.config.name,
                data_dir=str(self.config.data_dir)
            )

            # Get or compute statistics
            dataset_statistics = self._get_statistics(builder)

            # Construct dataset
            split = self._determine_split()
            dataset = dl.DLataset.from_rlds(
                builder,
                split=split,
                shuffle=self.config.shuffle,
                num_parallel_reads=self.config.num_parallel_reads
            )

            # Apply filters and transforms
            dataset = self._apply_filters(dataset)
            dataset = self._apply_transforms(dataset, dataset_statistics)

            return dataset, dataset_statistics

        except Exception as e:
            logger.error(f"Error building dataset: {str(e)}")
            raise

    def _get_statistics(self, builder: tfds.core.DatasetBuilder) -> Dict[str, Any]:
        """Get or compute dataset statistics."""
        if isinstance(self.config.dataset_statistics, str):
            with open(self.config.dataset_statistics) as f:
                return json.load(f)

        if self.config.dataset_statistics is not None:
            return self.config.dataset_statistics

        return self._compute_statistics(builder)

    def _compute_statistics(self, builder: tfds.core.DatasetBuilder) -> Dict[str, Any]:
        """Compute dataset statistics."""
        dataset = dl.DLataset.from_rlds(
            builder,
            split="all",
            shuffle=False
        )
        
        for filter_fn_spec in self.config.filter_functions:
            dataset = dataset.filter(ModuleSpec.instantiate(filter_fn_spec))
            
        if self.config.ignore_errors:
            dataset = dataset.ignore_errors()
            
        dataset = dataset.traj_map(self._restructure).filter(self._is_nonzero_length)
        
        return get_dataset_statistics(
            dataset,
            hash_dependencies=(
                str(builder.info),
                str(self.config.proprio_obs_key),
                ModuleSpec.to_string(self.config.standardize_fn)
                if self.config.standardize_fn is not None else "",
                *map(ModuleSpec.to_string, self.config.filter_functions)
            ),
            save_dir=builder.data_dir,
            force_recompute=self.config.force_recompute_dataset_statistics
        )

    def _determine_split(self) -> str:
        """Determine appropriate dataset split."""
        if self.config.dataset_size_limit is not None:
            if self.config.train:
                return f"train[:{self.config.dataset_size_limit}]"
            return "val"

        if "val" not in self.config.split_keys:
            return "train[:95%]" if self.config.train else "train[95%:]"
            
        return "train" if self.config.train else "val"

    def _apply_filters(self, dataset: dl.DLataset) -> dl.DLataset:
        """Apply dataset filters."""
        for filter_fn_spec in self.config.filter_functions:
            dataset = dataset.filter(ModuleSpec.instantiate(filter_fn_spec))
            
        if self.config.ignore_errors:
            dataset = dataset.ignore_errors()
            
        return dataset

    def _apply_transforms(self, 
                         dataset: dl.DLataset,
                         statistics: Dict[str, Any]) -> dl.DLataset:
        """Apply dataset transforms."""
        dataset = dataset.traj_map(
            self._restructure,
            self.config.num_parallel_calls
        ).filter(self._is_nonzero_length)

        if not self.config.skip_norm:
            dataset = dataset.traj_map(
                lambda x: normalize_action_and_proprio(
                    x,
                    metadata=statistics,
                    normalization_type=self.config.action_proprio_normalization_type
                ),
                self.config.num_parallel_calls
            )
        else:
            logger.warning("Dataset normalization turned off")

        return dataset

    def _restructure(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Restructure trajectory data."""
        if self.config.standardize_fn is not None:
            traj = ModuleSpec.instantiate(self.config.standardize_fn)(traj)

        traj_len = tf.shape(traj["action"])[0]
        obs = traj["observation"]
        new_obs = {}

        # Process images
        for new, old in self.config.image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"image_{new}"] = obs[old]

        # Process depth images
        for new, old in self.config.depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)
            else:
                new_obs[f"depth_{new}"] = obs[old]

        # Process proprio
        if self.config.proprio_obs_key is not None:
            new_obs["proprio"] = tf.cast(obs[self.config.proprio_obs_key], tf.float32)

        # Add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # Process language instruction
        task = {}
        if self.config.language_key is not None:
            task["language_instruction"] = sample_match_keys_uniform(
                traj,
                self.config.language_key
            )

        return {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(self.config.name, traj_len)
        }

    @staticmethod
    def _is_nonzero_length(traj: Dict[str, Any]) -> tf.Tensor:
        """Check if trajectory has non-zero length."""
        return tf.shape(traj["action"])[0] > 0


def make_dataset_from_rlds(
        name: str,
        data_dir: Union[str, Path],
        **kwargs
) -> Tuple[dl.DLataset, Dict[str, Any]]:
    """Create dataset from RLDS format."""
    config = DatasetConfig(name=name, data_dir=data_dir, **kwargs)
    builder = DatasetBuilder(config)
    return builder.build()


class MixtureBuilder:
    """Handles dataset mixture creation."""

    def __init__(self,
                 dataset_kwargs_list: Sequence[Dict[str, Any]],
                 sample_weights: Optional[Sequence[float]] = None,
                 shuffle_buffer_size: int = 10000,
                 balance_weights: bool = False,
                 traj_transform_threads: Optional[int] = None,
                 traj_read_threads: Optional[int] = None):
        self.dataset_kwargs_list = dataset_kwargs_list
        self.sample_weights = sample_weights or [1.0] * len(dataset_kwargs_list)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.balance_weights = balance_weights
        self.traj_transform_threads = traj_transform_threads
        self.traj_read_threads = traj_read_threads

    def build(self,
              traj_config: TrajectoryConfig,
              frame_config: FrameConfig) -> dl.DLataset:
        """Build interleaved dataset mixture."""
        try:
            # Validate weights
            if len(self.sample_weights) != len(self.dataset_kwargs_list):
                raise ValueError("Weights must match number of datasets")

            # Get dataset sizes and statistics
            dataset_sizes = []
            all_statistics = {}
            
            for kwargs in self.dataset_kwargs_list:
                _, statistics = make_dataset_from_rlds(**kwargs)
                dataset_sizes.append(statistics["num_transitions"])
                all_statistics[kwargs["name"]] = statistics

            # Process weights
            weights = np.array(self.sample_weights)
            if self.balance_weights:
                weights = weights * np.array(dataset_sizes)
            weights = weights / weights.sum()

            # Print mixture info
            pprint_data_mixture(self.dataset_kwargs_list, weights)

            # Allocate threads
            transform_threads = allocate_threads(
                self.traj_transform_threads,
                weights
            )
            read_threads = allocate_threads(
                self.traj_read_threads,
                weights
            )

            # Build datasets
            datasets = self._build_datasets(
                transform_threads,
                read_threads,
                all_statistics,
                traj_config
            )

            # Combine datasets
            dataset = dl.DLataset.sample_from_datasets(
                datasets,
                weights
            ).shuffle(self.shuffle_buffer_size)

            # Apply frame transforms
            dataset = obs_transforms.apply_frame_transforms(
                dataset,
                **vars(frame_config)
            )

            if traj_config.batch_size is not None:
                dataset = dataset.batch(traj_config.batch_size)

            dataset = dataset.with_ram_budget(1)
            dataset.dataset_statistics = all_statistics
            dataset.sample_weights = weights

            return dataset

        except Exception as e:
            logger.error(f"Error building dataset mixture: {str(e)}")
            raise

    def _build_datasets(self,
                       transform_threads: np.ndarray,
                       read_threads: np.ndarray,
                       statistics: Dict[str, Dict],
                       config: TrajectoryConfig) -> List[dl.DLataset]:
        """Build individual datasets in mixture."""
        datasets = []
        for kwargs, threads, reads in zip(
            self.dataset_kwargs_list,
            transform_threads,
            read_threads
        ):
            dataset, _ = make_dataset_from_rlds(
                **kwargs,
                num_parallel_calls=threads,
                num_parallel_reads=reads,
                dataset_statistics=statistics[kwargs["name"]]
            )
            
            dataset = traj_transforms.apply_trajectory_transforms(
                dataset.repeat(),
                **vars(config)
            ).flatten(num_parallel_calls=threads)
            
            datasets.append(dataset)
            
        return datasets


def make_interleaved_dataset(
        dataset_kwargs_list: Sequence[Dict[str, Any]],
        sample_weights: Optional[Sequence[float]] = None,
        **kwargs
) -> dl.DLataset:
    """Create interleaved dataset from multiple sources."""
    builder = MixtureBuilder(
        dataset_kwargs_list,
        sample_weights,
        **kwargs
    )
    return builder.build()


@dataclass
class SingleDatasetConfig:
    """Configuration for single dataset creation."""
    train: bool = True
    batch_size: Optional[int] = None
    traj_transform_kwargs: Dict[str, Any] = None
    frame_transform_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.traj_transform_kwargs is None:
            self.traj_transform_kwargs = {}
        if self.frame_transform_kwargs is None:
            self.frame_transform_kwargs = {}


class SingleDatasetBuilder:
    """Handles single dataset creation with configuration."""

    def __init__(self, dataset_kwargs: Dict[str, Any], config: SingleDatasetConfig):
        self.dataset_kwargs = dataset_kwargs
        self.config = config

    def build(self) -> dl.DLataset:
        """Build single dataset with transforms."""
        try:
            # Create base dataset
            dataset, dataset_statistics = make_dataset_from_rlds(
                **self.dataset_kwargs,
                # train=self.config.train
            )

            # Apply trajectory transforms
            dataset = traj_transforms.apply_trajectory_transforms(
                dataset,
                train=self.config.train,
                **self.config.traj_transform_kwargs
            )

            # Apply frame transforms
            dataset = obs_transforms.apply_frame_transforms(
                dataset,
                train=self.config.train,
                **self.config.frame_transform_kwargs
            )

            # Get length before potential batching
            dataset_len = dataset_statistics["num_transitions"]

            # Apply batching if configured
            if self.config.batch_size is not None:
                dataset = dataset.batch(self.config.batch_size)

            # Optimize memory usage
            dataset = dataset.with_ram_budget(1)

            # Add metadata
            dataset.dataset_statistics = dataset_statistics
            dataset.dataset_len = dataset_len

            return dataset

        except Exception as e:
            logger.error(
                f"Error building dataset {self.dataset_kwargs.get('name', 'unknown')}: {str(e)}"
            )
            raise


def make_single_dataset(
    dataset_kwargs: Dict[str, Any],
    config: Optional[SingleDatasetConfig] = None
) -> dl.DLataset:
    """Creates a single dataset with transforms.

    Args:
        dataset_kwargs: Keyword arguments for dataset creation
        config: Configuration for dataset creation and transforms
        
    Returns:
        Transformed dataset with statistics and length information
        
    Example:
        >>> config = SingleDatasetConfig(
        ...     train=True,
        ...     batch_size=32,
        ...     traj_transform_kwargs={"window_size": 2},
        ...     frame_transform_kwargs={"resize_size": (224, 224)}
        ... )
        >>> dataset = make_single_dataset(dataset_kwargs, config)
    """
    builder = SingleDatasetBuilder(
        dataset_kwargs,
        config or SingleDatasetConfig()
    )
    return builder.build()