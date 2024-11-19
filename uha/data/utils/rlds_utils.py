import os
import hashlib
from typing import Callable
import json

import tensorflow as tf
import logging
from typing import Mapping, Optional, Sequence, Tuple, Union
from functools import partial
import numpy as np
import tensorflow_datasets as tfds
import dlimp as dl

from uha.data.utils.spec import ModuleSpec
from uha.data.utils.data_utils import (
    sample_match_keys_uniform,

)
from uha.data.utils.dataset_index import DATASET_NAME_TO_INDEX

logger = logging.getLogger(__name__)


class RLDSProcessing:
    """Utility class for processing RLDS dataset components."""
    
    @staticmethod
    def process_observations(
        traj: dict,
        image_obs_keys: Mapping[str, Optional[str]],
        depth_obs_keys: Mapping[str, Optional[str]],
        proprio_obs_key: Optional[str]
    ) -> Tuple[dict, int]:
        """Process observation data and return new observations and trajectory length."""
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}

        # Process images
        for new, old in image_obs_keys.items():
            new_obs[f"image_{new}"] = (
                tf.repeat("", traj_len) if old is None else old_obs[old]
            )

        # Process depth images
        for new, old in depth_obs_keys.items():
            new_obs[f"depth_{new}"] = (
                tf.repeat("", traj_len) if old is None else old_obs[old]
            )

        # Add proprio if specified
        if proprio_obs_key is not None:
            new_obs["proprio"] = tf.cast(old_obs[proprio_obs_key], tf.float32)

        new_obs["timestep"] = tf.range(traj_len)
        return new_obs, traj_len

    @staticmethod
    def process_robot_information(old_obs: dict, traj_len: int) -> tf.Tensor:
        """Process robot information data."""
        try:
            robot_info = old_obs['robot_information']
            tf.debugging.assert_type(robot_info, tf.string, "robot_information must be string tensor")
            tf.debugging.assert_rank(robot_info, 1, "robot_information must be 1D tensor")
            return tf.tile(robot_info, [traj_len])
        except KeyError as e:
            raise KeyError(f"Missing required robot information field: {e}")
        
    # make a method based on the dataset name that adds an index as a new field called dataset_index
    # take the dataset name and get the index from from uha.data.utils.dataset_index import DATASET_NAME_TO_INDEX

    @staticmethod
    def process_dataset_index(old_obs: dict, traj_len: int, dataset_name: str) -> tf.Tensor:
        """
        Process dataset index data by converting dataset name to index and creating a tensor.
        
        Args:
            old_obs (dict): Original observation dictionary (unused but kept for consistency)
            traj_len (int): Length of trajectory to repeat index for
            dataset_name (str): Name of the dataset to get index for
            
        Returns:
            tf.Tensor: 1D int32 tensor of length traj_len containing the dataset index
            
        Raises:
            KeyError: If dataset_name is not found in DATASET_NAME_TO_INDEX mapping
        """
        print('-'*50)
        print("dataset_name", dataset_name)
        print('-'*50)
        try:
            # Get dataset index and convert to tensor
            dataset_index = DATASET_NAME_TO_INDEX[dataset_name]
            # Create constant tensor with proper type
            index_tensor = tf.constant(dataset_index, dtype=tf.int32)
            # Repeat for trajectory length
            return tf.repeat(index_tensor, repeats=[traj_len])
        except KeyError:
            raise KeyError(f"Dataset name '{dataset_name}' not found in index mapping")

    def process_frequency(old_obs: dict, traj_len: int) -> tf.Tensor:
        """Process frequency data from the trajectory.
        
        Args:
            old_obs: Original observation dictionary
            traj_len: Length of the trajectory
            
        Returns:
            tf.Tensor: Tiled frequency tensor matching trajectory length
            
        Raises:
            KeyError: If frequency field is missing
            ValueError: If frequency is not the expected type or shape
        """
        try:
            # Get frequency value
            frequency = old_obs['frequency']
            
            # Type and shape validation
            tf.debugging.assert_type(
                frequency, 
                tf.int32,
                message="frequency must be int32 tensor"
            )
            tf.debugging.assert_scalar(
                frequency,
                message="frequency must be a scalar value"
            )
            
            # Tile frequency to match trajectory length
            return tf.repeat(frequency, traj_len)
            
        except KeyError as e:
            raise KeyError(f"Missing required frequency field: {e}")
        except Exception as e:
            raise ValueError(f"Error processing frequency: {e}")

    @staticmethod
    def determine_language_key(
        path: str,
        use_gt: bool,
        use_nils: bool,
        language_key: str,
        language_key_NILS: str
    ) -> str:
        """Determine which language key to use based on path and conditions."""
        if use_gt:
            logger.info("Using GT")
            return language_key
        elif use_nils:
            return language_key_NILS
        return "ERROR"

    @staticmethod
    def process_language(
        traj: dict,
        traj_len: int,
        Gt_ann_dirs: Optional[list],
        NILS_ann_dirs: Optional[list],
        language_key: Optional[str],
        language_key_NILS: Optional[str]
    ) -> Tuple[dict, str]:
        """Process language data and return task dict and local language key."""
        task = {}
        
        # Handle standard language processing
        if not all([Gt_ann_dirs, NILS_ann_dirs]):
            local_lang_key = language_key
            if language_key is not None:
                task["language_instruction"] = sample_match_keys_uniform(traj, language_key)
            return task, local_lang_key

        path = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
        gt_pattern = "|".join([".*" + key + ".*" for key in Gt_ann_dirs])
        nils_pattern = "|".join([".*" + key + ".*" for key in NILS_ann_dirs])
        
        use_gt = tf.strings.regex_full_match(path, gt_pattern)
        use_nils = tf.strings.regex_full_match(path, nils_pattern)
        
        if len(Gt_ann_dirs) > 0 and Gt_ann_dirs[0] == "REMAINING" and not use_nils:
            use_gt, use_nils = True, False
            
        logger.info(f"GT: {use_gt}, NILS: {use_nils}")
        
        local_lang_key = RLDSProcessing.determine_language_key(
            path, use_gt, use_nils, language_key, language_key_NILS
        )
        task["language_instruction"] = sample_match_keys_uniform(
            traj, 
            language_key if use_gt else f"{language_key_NILS}*"
        )
        
        return task, local_lang_key

    @staticmethod
    def add_language_metadata(
        task: dict,
        local_lang_key: str,
        traj_len: int,
        language_key: Optional[str]
    ) -> None:
        """Add language-related metadata to task dict."""
        if language_key is not None:
            if local_lang_key == "ERROR":
                task["language_key"] = tf.repeat(-1, traj_len)
            elif tf.strings.regex_full_match(local_lang_key, ".*NILS.*"):
                task["language_key"] = tf.repeat(1, traj_len)
            else:
                task["language_key"] = tf.repeat(0, traj_len)

            if task["language_instruction"].dtype != tf.string:
                raise ValueError(
                    f"Language key {local_lang_key} has dtype {task['language_instruction'].dtype}, "
                    "but it must be tf.string."
                )


# dataset_utils.py
class DatasetUtils:
    """Utility class for dataset operations."""
    
    @staticmethod
    def get_dataset_split(
        builder: tfds.builder,
        train: bool,
        dataset_size_limit: Optional[int]
    ) -> str:
        """Determine the appropriate dataset split."""
        if dataset_size_limit is not None and isinstance(dataset_size_limit, int):
            if (train and 
                builder.info.splits["train"].num_examples >= 
                (dataset_size_limit + int(dataset_size_limit * 0.05))):
                return f"train[:{dataset_size_limit}]"
            elif train and "val" not in builder.info.splits:
                return "train[:95%]"
            elif train:
                return "train"
            elif not train and "val" not in builder.info.splits:
                return "train[95%:]"
            return "val"
        else:
            return ("train[:95%]" if train else "train[95%:]") if "val" not in builder.info.splits else ("train" if train else "val")

    @staticmethod
    def validate_normalization_mask(
        action_normalization_mask: Optional[Sequence[bool]],
        dataset_statistics: dict
    ) -> None:
        """Validate normalization mask against dataset statistics."""
        if action_normalization_mask is not None:
            if len(action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
                raise ValueError(
                    f"Normalization mask length ({len(action_normalization_mask)}) "
                    f"doesn't match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
                )
            dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)


def convert_scalar_to_1d(tensor, reference_tensor=None):
    """
    Converts a scalar tensor (shape ()) to a 1D tensor with shape (None,).
    If reference_tensor is provided, tile the scalar tensor to match the batch size.
    """
    if tf.rank(tensor) == 0:
        tensor = tf.expand_dims(tensor, axis=0)  # Expand scalar to (1,)
        tensor.set_shape([None])  # Set the shape to (None,)
    
    if reference_tensor is not None:
        ref_shape = tf.shape(reference_tensor)
        if tf.rank(tensor) == 1 and tf.shape(tensor)[0] == 1:
            # Tile to match the batch size of reference_tensor
            tensor = tf.tile(tensor, [ref_shape[0]])
    
    return tensor


class DatasetStatisticsComputer:
    """Handles computation and caching of dataset statistics."""
    
    @staticmethod
    def initialize_stats(data: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Initialize statistical accumulators with proper data types."""
        # Convert to float32 for numerical stability
        data = tf.cast(data, tf.float32)
        shape = data.shape[-1]
        
        return (
            tf.zeros(shape, dtype=tf.float32),  # sums
            tf.zeros(shape, dtype=tf.float32),  # sum_squares
            tf.ones(shape, dtype=tf.float32) * float('inf'),  # mins
            tf.ones(shape, dtype=tf.float32) * float('-inf')  # maxs
        )

    @staticmethod
    def update_running_stats(
        data: tf.Tensor,
        sums: tf.Tensor,
        sum_squares: tf.Tensor,
        mins: tf.Tensor,
        maxs: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, int]:
        """Update running statistics with proper type handling."""
        # Ensure data is float32
        data = tf.cast(data, tf.float32)
        flat_data = tf.reshape(data, [-1, data.shape[-1]])
        
        # Update statistics
        sums += tf.reduce_sum(flat_data, axis=0)
        sum_squares += tf.reduce_sum(tf.square(flat_data), axis=0)
        mins = tf.minimum(mins, tf.reduce_min(flat_data, axis=0))
        maxs = tf.maximum(maxs, tf.reduce_max(flat_data, axis=0))
        count = tf.shape(flat_data)[0]
        
        return sums, sum_squares, mins, maxs, count

    @staticmethod
    def compute_statistics(dataset: tf.data.Dataset) -> dict:
        """Compute mean, std, min, max for actions and proprioceptive data."""
        logger.info("Computing dataset statistics...")
        
        # Initialize storage for different keys (action, proprio)
        stats_storage = {}
        
        def process_trajectory(traj: dict) -> None:
            """Process a single trajectory, updating statistics."""
            # Handle action data
            if 'action' not in stats_storage:
                stats_storage['action'] = {
                    'sums': None,
                    'sum_squares': None,
                    'mins': None,
                    'maxs': None,
                    'count': 0
                }
                
                # Initialize with first action
                (stats_storage['action']['sums'],
                 stats_storage['action']['sum_squares'],
                 stats_storage['action']['mins'],
                 stats_storage['action']['maxs']) = DatasetStatisticsComputer.initialize_stats(traj['action'])
            
            # Update action statistics
            (stats_storage['action']['sums'],
             stats_storage['action']['sum_squares'],
             stats_storage['action']['mins'],
             stats_storage['action']['maxs'],
             count) = DatasetStatisticsComputer.update_running_stats(
                traj['action'],
                stats_storage['action']['sums'],
                stats_storage['action']['sum_squares'],
                stats_storage['action']['mins'],
                stats_storage['action']['maxs']
            )
            stats_storage['action']['count'] += count
            
            # Handle proprio data if present
            if 'proprio' in traj['observation']:
                if 'proprio' not in stats_storage:
                    stats_storage['proprio'] = {
                        'sums': None,
                        'sum_squares': None,
                        'mins': None,
                        'maxs': None,
                        'count': 0
                    }
                    
                    # Initialize with first proprio data
                    (stats_storage['proprio']['sums'],
                     stats_storage['proprio']['sum_squares'],
                     stats_storage['proprio']['mins'],
                     stats_storage['proprio']['maxs']) = DatasetStatisticsComputer.initialize_stats(
                        traj['observation']['proprio']
                    )
                
                # Update proprio statistics
                (stats_storage['proprio']['sums'],
                 stats_storage['proprio']['sum_squares'],
                 stats_storage['proprio']['mins'],
                 stats_storage['proprio']['maxs'],
                 count) = DatasetStatisticsComputer.update_running_stats(
                    traj['observation']['proprio'],
                    stats_storage['proprio']['sums'],
                    stats_storage['proprio']['sum_squares'],
                    stats_storage['proprio']['mins'],
                    stats_storage['proprio']['maxs']
                )
                stats_storage['proprio']['count'] += count

        # Process all trajectories
        for traj in dataset:
            process_trajectory(traj)

        # Compute final statistics
        statistics = {}
        for key, stats in stats_storage.items():
            mean = stats['sums'] / tf.cast(stats['count'], tf.float32)
            variance = (stats['sum_squares'] / tf.cast(stats['count'], tf.float32)) - tf.square(mean)
            std = tf.sqrt(tf.maximum(variance, 1e-10))
            
            statistics[key] = {
                'mean': mean.numpy(),
                'std': std.numpy(),
                'min': stats['mins'].numpy(),
                'max': stats['maxs'].numpy(),
            }
        
        return statistics

    @staticmethod
    def get_cache_path(save_dir: str, hash_str: str) -> str:
        """Generate cache file path for dataset statistics."""
        cache_dir = os.path.join(save_dir, "statistics_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"stats_{hash_str}.json")

    @staticmethod
    def load_or_compute_statistics(
        dataset: tf.data.Dataset,
        save_dir: str,
        hash_dependencies: Tuple[str, ...],
        force_recompute: bool = False
    ) -> dict:
        """Load cached statistics or compute new ones."""
        # Create hash from dependencies
        hash_str = hashlib.sha256(
            "".join(str(d) for d in hash_dependencies).encode()
        ).hexdigest()
        
        cache_path = DatasetStatisticsComputer.get_cache_path(save_dir, hash_str)
        
        # Try to load cached statistics
        if not force_recompute and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    logger.info(f"Loaded cached statistics from {cache_path}")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached statistics: {e}")

        # Compute new statistics
        statistics = DatasetStatisticsComputer.compute_statistics(dataset)
        
        # Cache the results
        try:
            with open(cache_path, 'w') as f:
                json.dump(statistics, f)
            logger.info(f"Cached statistics to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache statistics: {e}")
            
        return statistics


def compute_dataset_statistics(
    builder: tfds.builder,
    filter_functions: Sequence[ModuleSpec],
    ignore_errors: bool,
    restructure_fn: Callable,
    proprio_obs_key: Optional[str],
    standardize_fn: Optional[ModuleSpec],
    force_recompute: bool = False
) -> dict:
    """Compute or load cached dataset statistics."""
    # Create full dataset for statistics computation
    full_dataset = dl.DLataset.from_rlds(
        builder, 
        split="all", 
        shuffle=False
    )
    
    # Apply filters
    for filter_fcn_spec in filter_functions:
        full_dataset = full_dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
    
    if ignore_errors:
        full_dataset = full_dataset.ignore_errors()
        
    # Apply restructuring
    full_dataset = full_dataset.traj_map(restructure_fn).filter(
        lambda traj: tf.shape(traj["action"])[0] > 0
    )
    
    # Generate hash dependencies
    hash_dependencies = (
        str(builder.info),
        str(proprio_obs_key),
        ModuleSpec.to_string(standardize_fn) if standardize_fn is not None else "",
        *map(ModuleSpec.to_string, filter_functions),
    )
    
    # Create unique hash
    unique_hash = hashlib.sha256(
        "".join(hash_dependencies).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()

    # Set up cache paths
    local_path = os.path.expanduser(
        os.path.join(
            "~",
            ".cache",
            "octo",
            f"dataset_statistics_{unique_hash}.json",
        )
    )

    path = tf.io.gfile.join(builder.data_dir, f"dataset_statistics_{unique_hash}.json") \
        if builder.data_dir is not None else local_path

    # Try loading from cache
    if not force_recompute:
        if tf.io.gfile.exists(path):
            logging.info(f"Loading existing dataset statistics from {path}.")
            with tf.io.gfile.GFile(path, "r") as f:
                return json.load(f)
        if os.path.exists(local_path):
            logging.info(f"Loading existing dataset statistics from {local_path}.")
            with open(local_path, "r") as f:
                return json.load(f)

    # If we get here, need to compute statistics
    cardinality = full_dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    logging.info("Computing dataset statistics. This may take awhile...")
    
    actions = []
    proprios = []
    num_transitions = 0
    num_trajectories = 0
    
    # Process each trajectory
    for traj in tqdm.tqdm(
        full_dataset.iterator(),
        total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
    ):
        actions.append(traj["action"])
        if "proprio" in traj["observation"]:
            proprios.append(traj["observation"]["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1

    # Compute statistics
    actions = np.concatenate(actions)
    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
            "p99": np.quantile(actions, 0.99, 0).tolist(),
            "p01": np.quantile(actions, 0.01, 0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }

    if proprios:
        proprios = np.concatenate(proprios)
        metadata["proprio"] = {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
            "p99": np.quantile(proprios, 0.99, 0).tolist(),
            "p01": np.quantile(proprios, 0.01, 0).tolist(),
        }

    # Try to cache the results
    try:
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        logging.warning(
            f"Could not write dataset statistics to {path}. "
            f"Writing to {local_path} instead."
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(metadata, f)

    return metadata

def combine_dataset_statistics(
    all_dataset_statistics: Sequence[dict],
) -> dict:
    """Merges dataset statistics from multiple datasets."""
    merge_stat_keys = ["action", "proprio"]

    num_trajectories = [stat["num_trajectories"] for stat in all_dataset_statistics]
    num_transitions = [stat["num_transitions"] for stat in all_dataset_statistics]
    stat_weights = [
        transitions / sum(num_transitions) for transitions in num_transitions
    ]

    combined_dataset_statistics = {}
    for key in merge_stat_keys:
        if not all(key in stat for stat in all_dataset_statistics):
            continue
            
        # Combine all statistics
        combined_dataset_statistics[key] = {
            "min": np.array([stat[key]["min"] for stat in all_dataset_statistics])
            .min(0)
            .tolist(),
            "max": np.array([stat[key]["max"] for stat in all_dataset_statistics])
            .max(0)
            .tolist(),
            "p01": np.array([stat[key]["p01"] for stat in all_dataset_statistics])
            .min(0)
            .tolist(),
            "p99": np.array([stat[key]["p99"] for stat in all_dataset_statistics])
            .max(0)
            .tolist(),
            "mean": np.array([
                stat[key]["mean"] * w
                for stat, w in zip(all_dataset_statistics, stat_weights)
            ]).sum(0).tolist(),
            "std": np.sqrt(
                np.array([
                    n * np.array(stat[key]["std"]) ** 2
                    + n * (np.array(stat[key]["mean"]) - np.array(combined_dataset_statistics[key]["mean"])) ** 2
                    for stat, n in zip(all_dataset_statistics, num_transitions)
                ]).sum(0)
                / sum(num_transitions)
            ).tolist()
        }

    combined_dataset_statistics["num_trajectories"] = num_trajectories
    combined_dataset_statistics["num_transitions"] = num_transitions
    return combined_dataset_statistics