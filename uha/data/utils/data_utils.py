from dataclasses import dataclass

from fnmatch import fnmatch
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import hashlib
import json
import logging
import os
from copy import copy
import dlimp as dl
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
# import 
import tensorflow_probability as tfp

logger = logging.getLogger(__name__)


class NormalizationType(str, Enum):
    """Normalization schemes for action and proprio data"""
    NORMAL = "normal"  # Normalize to mean 0, std 1
    BOUNDS = "bounds"  # Normalize to [-1, 1]


@dataclass
class DatasetStatistics:
    """Dataset statistics container with validation"""
    mean: np.ndarray
    std: np.ndarray
    max_val: np.ndarray
    min_val: np.ndarray
    p99: np.ndarray
    p01: np.ndarray
    
    @classmethod
    def from_dict(cls, data: Dict[str, List[float]]) -> 'DatasetStatistics':
        """Create statistics from dictionary"""
        return cls(
            mean=np.array(data['mean']),
            std=np.array(data['std']),
            max_val=np.array(data['max']),
            min_val=np.array(data['min']),
            p99=np.array(data['p99']),
            p01=np.array(data['p01'])
        )
    
    def to_dict(self) -> Dict[str, List[float]]:
        """Convert statistics to dictionary"""
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'max': self.max_val.tolist(),
            'min': self.min_val.tolist(),
            'p99': self.p99.tolist(),
            'p01': self.p01.tolist()
        }


class DataProcessor:
    """Helper class for data processing operations"""
    
    @staticmethod
    def tree_map(fn: Callable, tree: Dict) -> Dict:
        """Maps a function over a nested dictionary"""
        return {
            k: DataProcessor.tree_map(fn, v) if isinstance(v, dict) else fn(v) 
            for k, v in tree.items()
        }

    @staticmethod
    def tree_merge(*trees: Dict) -> Dict:
        """Merges a list of nested dictionaries"""
        merged = {}
        for tree in trees:
            for k, v in tree.items():
                if isinstance(v, dict):
                    merged[k] = DataProcessor.tree_merge(merged.get(k, {}), v)
                else:
                    merged[k] = v
        return merged


class PaddingGenerator:
    """Handles creation of padding tensors"""
    
    @staticmethod
    def create_padding(tensor: tf.Tensor) -> tf.Tensor:
        """Create appropriate padding based on tensor type"""
        if tf.debugging.is_numeric_tensor(tensor):
            return tf.zeros_like(tensor)
        elif tensor.dtype == tf.string:
            return tf.fill(tf.shape(tensor), "")
        raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}")


class DatasetUtils:
    """Core dataset utility functions"""
    
    @staticmethod
    @lru_cache(maxsize=128)
    def get_dataset_hash(dependencies: Tuple[str, ...]) -> str:
        """Generate unique hash for dataset configuration"""
        return hashlib.sha256(
            "".join(dependencies).encode("utf-8"),
            usedforsecurity=False
        ).hexdigest()

    @staticmethod
    def get_statistics_path(save_dir: Optional[str], unique_hash: str) -> Path:
        """Get appropriate path for statistics file"""
        if save_dir:
            return Path(save_dir) / f"dataset_statistics_{unique_hash}.json"
        return Path.home() / ".cache" / "octo" / f"dataset_statistics_{unique_hash}.json"

    @classmethod
    def load_or_compute_statistics(
            cls,
            dataset: dl.DLataset,
            hash_dependencies: Tuple[str, ...],
            save_dir: Optional[str] = None,
            force_recompute: bool = False
    ) -> Dict[str, Any]:
        """Load or compute dataset statistics with caching"""
        unique_hash = cls.get_dataset_hash(hash_dependencies)
        stats_path = cls.get_statistics_path(save_dir, unique_hash)

        if not force_recompute:
            stats = cls._try_load_statistics(stats_path)
            if stats:
                return stats

        return cls._compute_and_save_statistics(dataset, stats_path)

    @staticmethod
    def _try_load_statistics(path: Path) -> Optional[Dict]:
        """Try to load statistics from file"""
        try:
            if path.exists():
                logger.info(f"Loading existing dataset statistics from {path}")
                return json.loads(path.read_text())
        except Exception as e:
            logger.warning(f"Failed to load statistics: {e}")
        return None

    @staticmethod
    def _compute_and_save_statistics(dataset: dl.DLataset, save_path: Path) -> Dict[str, Any]:
        """Compute and save dataset statistics"""
        logger.info("Computing dataset statistics (this may take a while)")
        
        actions = []
        proprios = []
        num_transitions = 0
        num_trajectories = 0

        for traj in tqdm.tqdm(dataset.iterator()):
            actions.append(traj["action"])
            if "proprio" in traj:
                proprios.append(traj["proprio"])
            num_transitions += traj["action"].shape[0]
            num_trajectories += 1

        actions = np.concatenate(actions)
        stats = {
            "action": DatasetStatistics(
                mean=actions.mean(0),
                std=actions.std(0),
                max_val=actions.max(0),
                min_val=actions.min(0),
                p99=np.quantile(actions, 0.99, 0),
                p01=np.quantile(actions, 0.01, 0)
            ).to_dict(),
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

        if proprios:
            proprios = np.concatenate(proprios)
            stats["proprio"] = DatasetStatistics(
                mean=proprios.mean(0),
                std=proprios.std(0),
                max_val=proprios.max(0),
                min_val=proprios.min(0),
                p99=np.quantile(proprios, 0.99, 0),
                p01=np.quantile(proprios, 0.01, 0)
            ).to_dict()

        # Save statistics
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(stats, indent=2))
        
        return stats


class Normalizer:
    """Handles data normalization operations"""
    
    @staticmethod
    def normalize_data(
            traj: Dict,
            metadata: Dict[str, Any],
            normalization_type: NormalizationType,
            action_mask: Optional[np.ndarray] = None
    ) -> Dict:
        """Normalize action and proprio data"""
        keys_to_normalize = {
            "action": "action",
        }
        if "proprio" in traj["observation"]:
            keys_to_normalize["proprio"] = "observation/proprio"

        if normalization_type == NormalizationType.NORMAL:
            return Normalizer._normalize_standard(traj, metadata, keys_to_normalize, action_mask)
        elif normalization_type == NormalizationType.BOUNDS:
            return Normalizer._normalize_bounds(traj, metadata, keys_to_normalize, action_mask)
        
        raise ValueError(f"Unknown normalization type {normalization_type}")

    @staticmethod
    def _normalize_standard(traj: Dict, metadata: Dict, keys: Dict[str, str], 
                          action_mask: Optional[np.ndarray]) -> Dict:
        """Normalize to mean 0, std 1"""
        for key, traj_key in keys.items():
            mask = action_mask if key == "action" and action_mask is not None else \
                   metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
            
            def normalize_fn(x):
                return tf.where(
                    mask,
                    (x - metadata[key]["mean"]) / (metadata[key]["std"] + 1e-8),
                    x
                )
            
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=normalize_fn
            )
        return traj

    @staticmethod
    def _normalize_bounds(traj: Dict, metadata: Dict, keys: Dict[str, str], 
                         action_mask: Optional[np.ndarray]) -> Dict:
        """Normalize to [-1, 1]"""
        for key, traj_key in keys.items():
            mask = action_mask if key == "action" and action_mask is not None else \
                   metadata[key].get("mask", tf.ones_like(metadata[key]["p01"], dtype=tf.bool))
            
            def normalize_fn(x):
                return tf.where(
                    mask,
                    tf.clip_by_value(
                        2 * (x - metadata[key]["p01"]) /
                        (metadata[key]["p99"] - metadata[key]["p01"] + 1e-8) - 1,
                        -1, 1
                    ),
                    x
                )
            
            traj = dl.transforms.selective_tree_map(
                traj,
                match=lambda k, _: k == traj_key,
                map_fn=normalize_fn
            )
        return traj


def binarize_gripper_actions(actions: tf.Tensor, 
                            open_boundary: float = 0.95,
                            close_boundary: float = 0.05) -> tf.Tensor:
    """Convert gripper actions from continuous to binary values"""
    open_mask = actions > open_boundary
    closed_mask = actions < close_boundary
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, actions.dtype)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, actions.dtype),
            lambda: is_open_float[i]
        )

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


class ThreadAllocator:
    """Manages thread allocation for dataset processing"""
    
    @staticmethod
    def allocate_threads(n: Optional[int], weights: np.ndarray) -> np.ndarray:
        """Allocate threads across datasets based on weights"""
        if n is None:
            return np.array([tf.data.AUTOTUNE] * len(weights))

        if len(weights) > n:
            raise ValueError("Number of threads must be >= number of weights")
            
        if not np.all(weights >= 0):
            raise ValueError("Weights must be non-negative")

        weights = np.array(weights) / np.sum(weights)
        allocation = np.zeros_like(weights, dtype=int)

        # Assign minimum threads
        while True:
            mask = (weights * n < 1) & (weights > 0)
            if not mask.any():
                break
            n -= mask.sum()
            allocation += mask.astype(int)
            weights[mask] = 0
            if weights.sum() > 0:
                weights = weights / weights.sum()

        # Allocate remaining threads
        fractional, integral = np.modf(weights * n)
        allocation += integral.astype(int)
        n -= integral.sum()
        
        # Distribute remaining threads
        for i in np.argsort(fractional)[::-1][:int(n)]:
            allocation[i] += 1

        return allocation
    

def get_dataset_statistics(
    dataset: dl.DLataset,
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Get or compute dataset statistics."""
    try:
        # Get statistics path
        unique_hash = hashlib.sha256(
            "".join(hash_dependencies).encode("utf-8"),
            usedforsecurity=False
        ).hexdigest()

        local_path = os.path.expanduser(
            os.path.join("~", ".cache", "octo", f"dataset_statistics_{unique_hash}.json")
        )
        
        path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json") if save_dir else local_path

        # Try to load cached statistics
        if not force_recompute:
            try:
                if tf.io.gfile.exists(path):
                    logger.info(f"Loading existing dataset statistics from {path}.")
                    with tf.io.gfile.GFile(path, "r") as f:
                        stats = json.load(f)
                        # Convert loaded lists to numpy arrays
                        for key in ["action", "proprio"]:
                            if key in stats:
                                for stat_key in ["mean", "std", "max", "min", "p99", "p01"]:
                                    if stat_key in stats[key]:
                                        stats[key][stat_key] = tf.constant(
                                            stats[key][stat_key], 
                                            dtype=tf.float32
                                        )
                        return stats
            except Exception as e:
                logger.warning(f"Error loading statistics from {path}: {e}")

        # Compute statistics
        logger.info("Computing dataset statistics...")
        actions = []
        proprios = []
        num_transitions = 0
        num_trajectories = 0

        # Use tqdm from auto
        for traj in tqdm(
            dataset.iterator(),
            desc="Computing statistics",
            unit="trajectories"
        ):
            actions.append(traj["action"])
            if "proprio" in traj:
                proprios.append(traj["proprio"])
            num_transitions += traj["action"].shape[0]
            num_trajectories += 1

        if not actions:
            raise ValueError("No trajectories found in dataset")

        # Compute statistics using TensorFlow
        actions = tf.concat(actions, axis=0)
        metadata = {
            "action": {
                "mean": tf.reduce_mean(actions, axis=0),
                "std": tf.math.reduce_std(actions, axis=0),
                "max": tf.reduce_max(actions, axis=0),
                "min": tf.reduce_min(actions, axis=0),
                "p99": tfp.stats.percentile(actions, 99.0, axis=0),
                "p01": tfp.stats.percentile(actions, 1.0, axis=0),
            },
            "num_transitions": int(num_transitions),
            "num_trajectories": int(num_trajectories),
        }

        if proprios:
            proprios = tf.concat(proprios, axis=0)
            metadata["proprio"] = {
                "mean": tf.reduce_mean(proprios, axis=0),
                "std": tf.math.reduce_std(proprios, axis=0),
                "max": tf.reduce_max(proprios, axis=0),
                "min": tf.reduce_min(proprios, axis=0),
                "p99": tfp.stats.percentile(proprios, 99.0, axis=0),
                "p01": tfp.stats.percentile(proprios, 1.0, axis=0),
            }

        # Save statistics
        try:
            # Convert tensors to lists for JSON serialization
            json_metadata = copy.deepcopy(metadata)
            for key in ["action", "proprio"]:
                if key in json_metadata:
                    for stat_key in ["mean", "std", "max", "min", "p99", "p01"]:
                        if stat_key in json_metadata[key]:
                            json_metadata[key][stat_key] = json_metadata[key][stat_key].numpy().tolist()
                            
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with tf.io.gfile.GFile(path, "w") as f:
                json.dump(json_metadata, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Could not save statistics to {path}: {e}")

        return metadata

    except Exception as e:
        logger.error(f"Error in get_dataset_statistics: {str(e)}")
        raise
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
        combined_mean = np.array(
            [
                stat[key]["mean"] * w
                for stat, w in zip(all_dataset_statistics, stat_weights)
            ]
        ).sum(0)
        # compute combined_std for denominator `n` instead of `n-1` since numpy uses that by default for std
        # https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
        combined_std = np.sqrt(
            np.array(
                [
                    n * np.array(stat[key]["std"]) ** 2
                    + n * (np.array(stat[key]["mean"]) - combined_mean) ** 2
                    for stat, n in zip(all_dataset_statistics, num_transitions)
                ]
            ).sum(0)
            / sum(num_transitions)
        )
        combined_dataset_statistics[key] = {
            "min": np.array([stat[key]["min"] for stat in all_dataset_statistics])
            .min(0)
            .tolist(),
            "max": np.array([stat[key]["max"] for stat in all_dataset_statistics])
            .max(0)
            .tolist(),
            "mean": combined_mean.tolist(),
            "std": combined_std.tolist(),
        }

    combined_dataset_statistics["num_trajectories"] = num_trajectories
    combined_dataset_statistics["num_transitions"] = num_transitions
    return combined_dataset_statistics


def pprint_data_mixture(
    dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]
) -> None:
    print(
        "\n######################################################################################"
    )
    print(
        f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #"
    )
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print(
        "######################################################################################\n"
    )


def sample_match_keys_uniform(d: dict, key_template: str):
    """Samples uniformly from all keys fnmatching the template."""
    match_keys = [key for key in d.keys() if fnmatch(key, key_template)]
    if not match_keys:
        raise ValueError(f"No matching key found for {key_template}. Keys: {d.keys()}")
    logging.info(f"Sampling uniformly across keys: {match_keys}")
    if len(match_keys) > 1:
        stacked = tf.stack([d[key] for key in match_keys])
        idx = tf.random.uniform((), 0, len(stacked) - 1, dtype=tf.int32)
        return stacked[idx]
    else:
        return d[match_keys[0]]


def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()
    }


def normalize_action_and_proprio(
    traj: Dict[str, Any],
    metadata: Dict[str, Any],
    normalization_type: NormalizationType
) -> Dict[str, Any]:
    """Normalize action and proprio data using Normalizer class."""
    try:
        return Normalizer.normalize_data(
            traj=traj,
            metadata=metadata,
            normalization_type=normalization_type
        )
    except Exception as e:
        logger.error(f"Error in normalize_action_and_proprio: {str(e)}")
        raise


def tree_merge(*trees: dict) -> dict:
    """Merges a list of nested dictionaries, with later dictionaries overriding earlier ones.
    
    Args:
        *trees: Variable number of dictionaries to merge
        
    Returns:
        dict: Merged dictionary with nested structure preserved
        
    Example:
        >>> d1 = {'a': 1, 'b': {'c': 2}}
        >>> d2 = {'b': {'d': 3}}
        >>> tree_merge(d1, d2)
        {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    merged = {}
    for tree in trees:
        for key, value in tree.items():
            if isinstance(value, dict):
                # If value is a dict, recursively merge with existing dict at key
                if key in merged:
                    merged[key] = tree_merge(merged[key], value) 
                else:
                    merged[key] = value
            else:
                # For non-dict values, later values override earlier ones
                merged[key] = value
    return merged

# For backwards compatibility
def merge_dictionaries(*dicts):
    """Alias for tree_merge for backwards compatibility"""
    return tree_merge(*dicts)


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    """Creates an appropriate padding tensor matching the input tensor's shape and type.
    
    Args:
        tensor: Input tensor to create padding for
        
    Returns:
        tf.Tensor: Padding tensor with same shape as input but filled with zeros or empty strings
        
    Example:
        >>> # For numeric tensors, returns zeros
        >>> to_padding(tf.ones([2,3]))
        <tf.Tensor: shape=(2, 3), dtype=float32, numpy=array([[0., 0., 0.], [0., 0., 0.]])>
        
        >>> # For string tensors, returns empty strings
        >>> to_padding(tf.constant(['a', 'b']))
        <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'', b''])>
    
    Raises:
        ValueError: If tensor type is not numeric or string
    """
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    elif tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    else:
        raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")
    

def allocate_threads(n: Optional[int], weights: np.ndarray) -> np.ndarray:
    """Allocates an integer number of threads across datasets based on weights.
    The final array sums to `n`, but each element is no less than 1.
    If `n` is None, then every dataset is assigned a value of AUTOTUNE.

    Args:
        n: Total number of threads to allocate, or None for AUTOTUNE
        weights: Array of weights for each dataset
        
    Returns:
        np.ndarray: Array of thread counts for each dataset
        
    Example:
        >>> allocate_threads(10, np.array([0.7, 0.3]))
        array([7, 3])
        >>> allocate_threads(None, np.array([0.5, 0.5]))
        array([AUTOTUNE, AUTOTUNE])
        
    Raises:
        ValueError: If weights are negative or n is less than number of weights
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert len(weights) <= n, "Number of threads must be >= length of weights"
    
    weights = np.array(weights) / np.sum(weights)
    allocation = np.zeros_like(weights, dtype=int)

    # First ensure minimum of 1 thread per dataset with nonzero weight
    while True:
        # Give remaining elements that would get less than 1 thread a single thread
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)
        # Recompute distribution over remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()

    # Allocate remaining threads proportionally 
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    
    # Distribute any remaining threads to datasets with largest fractional parts
    for i in np.argsort(fractional)[::-1][:int(n)]:
        allocation[i] += 1

    return allocation


def invert_gripper_actions(actions: tf.Tensor):
    return 1 - actions


def relabel_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels the actions to use the reached proprio instead. Discards the last timestep of the
    trajectory (since we don't have a next state to compute the action.)
    """
    # relabel the first 6 action dims (xyz position, xyz rotation) using the reached proprio
    movement_actions = (
        traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    )

    # discard the last timestep of the trajectory
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)

    # recombine to get full actions
    traj_truncated["action"] = tf.concat(
        [movement_actions, traj["action"][:-1, -1:]],
        axis=1,
    )

    return traj_truncated


def binarize_gripper_actions(actions: tf.Tensor, open_boundary: float = 0.95, close_boundary: float = 0.05) -> tf.Tensor:
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near
    0.0). As it transitions between the two, it sometimes passes through a few intermediate values. We relabel
    those intermediate values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel
    that chunk of intermediate values as the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > open_boundary # 0.95
    closed_mask = actions < close_boundary # 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, actions.dtype)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, actions.dtype),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )
    return new_actions


def rel_open_or_closed(actions: tf.Tensor):
    """
    Returns the initial absolute gripper state, given relative actions (-1 for closing, +1 for opening)
    Returns 1 if the gripper is initially open, 0 if it is initially closed.
    If nothing taken, assumes gripper is initially open.

    """
    opening_mask = actions > 1e-3
    closing_mask = actions < -1e-3
    old_state_mask = tf.where(opening_mask, -1, tf.where(closing_mask, -1, 0))
    # old_state_mask is 1 if closing, -1 if opening, 0 if no change

    def scan_fn(carry, i):
        return tf.cond(
            old_state_mask[i] == 0,
            lambda: tf.cast(carry, tf.float32),
            lambda: (tf.cast(old_state_mask[i], tf.float32) + 1) / 2,
        )

    return tf.scan(
        scan_fn,
        tf.range(tf.shape(actions)[0]),
        tf.zeros_like(actions[-1]),
        reverse=True,
    )[0]


def rel2abs_gripper_actions(actions: tf.Tensor):
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper actions
    (0 for closed, 1 for open). Assumes that the first relative gripper is not redundant
    (i.e. close when already closed).
    """
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1

    # -1 for closing, 1 for opening, 0 for no change
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(
            thresholded_actions[i] == 0,
            lambda: carry,
            lambda: thresholded_actions[i],
        )

    # if no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)
    # -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)

    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5
    return new_actions


def invert_gripper_actions(actions: tf.Tensor):
    return 1 - actions


def relabel_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels the actions to use the reached proprio instead. Discards the last timestep of the
    trajectory (since we don't have a next state to compute the action.)
    """
    # relabel the first 6 action dims (xyz position, xyz rotation) using the reached proprio
    movement_actions = (
        traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    )

    # discard the last timestep of the trajectory
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)

    # recombine to get full actions
    traj_truncated["action"] = tf.concat(
        [movement_actions, traj["action"][:-1, -1:]],
        axis=1,
    )

    return traj_truncated


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """Allocates an integer number of threads across datasets based on weights. The final array sums to `n`,
    but each element is no less than 1. If `n` is None, then every dataset is assigned a value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert (
        len(weights) <= n
    ), "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)
        # recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()
    # allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1
    return allocation

def filter_by_language_key(traj: Dict[str, Any], *, language_key_template: str) -> tf.Tensor:
    """Filter trajectory based on presence of non-empty language instruction.

    Args:
        traj: Trajectory dictionary containing language annotations
        language_key_template: Template for matching language key names, supports wildcards
        
    Returns:
        Boolean tensor indicating if trajectory should be kept

    Example:
        >>> filter_by_language_key(traj, language_key_template="language_instruction*")
    """
    match_keys = [key for key in traj.keys() if fnmatch(key, language_key_template)]
    if len(match_keys) == 0:
        raise ValueError(
            f"No matching key found for {language_key_template}. Keys: {traj.keys()}"
        )
    
    # Stack all matching language labels
    labels = tf.stack([traj[key] for key in match_keys], axis=0)
    
    # Keep trajectory if any label in any step is not empty
    return tf.math.reduce_any(labels != "")


def filter_by_task(traj: Dict[str, Any], *,
                  task_templates: List[str],
                  negative_task_templates: List[str]) -> tf.Tensor:
    """Filter trajectory based on task templates.

    Args:
        traj: Trajectory dictionary containing metadata
        task_templates: List of task patterns to match
        negative_task_templates: List of task patterns to exclude
        
    Returns:
        Boolean tensor indicating if trajectory should be kept
    """
    metadata = traj["episode_metadata"]["file_path"]
    
    # Check for positive matches
    pos_match = tf.math.reduce_any(
        [task_template in metadata for task_template in task_templates]
    )
    
    # Check for negative matches
    neg_match = tf.math.reduce_any(
        [task_template in metadata for task_template in negative_task_templates]
    )
    
    # Keep if positively matched and not negatively matched
    return pos_match and not neg_match


def filter_by_task_and_language_key(
    traj: Dict[str, Any], *,
    language_key_gt_template: str,
    language_key_NILS_template: str,
    gt_task_templates: List[str],
    NILS_task_templates: List[str],
    negative_task_templates: List[str]
) -> tf.Tensor:
    """Filter trajectory based on both task and language annotations.

    Args:
        traj: Trajectory dictionary
        language_key_gt_template: Template for ground truth language key
        language_key_NILS_template: Template for NILS language key
        gt_task_templates: List of ground truth task patterns
        NILS_task_templates: List of NILS task patterns
        negative_task_templates: List of patterns to exclude
        
    Returns:
        Boolean tensor indicating if trajectory should be kept
    """
    metadata = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
    logger.info(f"metadata: {metadata}")
    logger.info(traj[language_key_gt_template])

    # Check language annotations
    has_gt_annotation = tf.strings.length(traj[language_key_gt_template][0]) > 0
    has_NILS_annotation = tf.strings.length(traj[language_key_NILS_template][0]) > 0

    # Handle special "REMAINING" case
    if len(gt_task_templates) > 0 and gt_task_templates[0] == "REMAINING":
        pos_match_gt = True
    else:
        # Create regex pattern for ground truth tasks
        pos_pattern = '|'.join([".*" + template + ".*" for template in gt_task_templates])
        pos_match_gt = tf.reduce_any(tf.strings.regex_full_match(metadata, pos_pattern))

    # Match NILS tasks
    if len(NILS_task_templates) > 0:
        pos_pattern_NILS = '|'.join([".*" + template + ".*" for template in NILS_task_templates])
        pos_match_NILS = tf.strings.regex_full_match(metadata, pos_pattern_NILS)
        logger.info(f"pos_match_NILS: {pos_match_NILS}")
    else:
        pos_match_NILS = False

    # Match negative templates
    if len(negative_task_templates) > 0:
        neg_pattern = '|'.join([".*" + template + ".*" for template in negative_task_templates])
        neg_match = tf.reduce_any(tf.strings.regex_full_match(metadata, neg_pattern))
    else:
        neg_match = False

    # Keep if:
    # - Has ground truth annotation and matches ground truth tasks OR
    # - Has NILS annotation and matches NILS tasks
    # AND doesn't match negative patterns
    valid_traj = ((pos_match_gt and has_gt_annotation) or 
                 (pos_match_NILS and has_NILS_annotation)) and not neg_match
    
    return tf.cast(valid_traj, tf.bool)


def sample_match_keys_uniform(d: Dict[str, Any], key_template: str) -> Any:
    """Sample uniformly from all keys matching the template.

    Args:
        d: Dictionary to sample from 
        key_template: Template for matching keys, supports wildcards
        
    Returns:
        Value from randomly selected matching key

    Raises:
        ValueError: If no keys match the template
    """
    match_keys = [key for key in d.keys() if fnmatch(key, key_template)]
    if not match_keys:
        raise ValueError(
            f"No matching key found for {key_template}. Keys: {d.keys()}"
        )
        
    logging.info(f"Sampling uniformly across keys: {match_keys}")
    
    if len(match_keys) > 1:
        # Stack values from all matching keys
        stacked = tf.stack([d[key] for key in match_keys])
        # Sample random index
        idx = tf.random.uniform((), 0, len(stacked) - 1, dtype=tf.int32)
        return stacked[idx]
    else:
        return d[match_keys[0]]
    

import logging
from typing import Any

log = logging.getLogger(__name__)

def hydra_locate(path: str) -> Any:
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    
    Args:
        path: Dotted path to the object (e.g. "torch.float32")
        
    Returns:
        The requested object
        
    Raises:
        ImportError: If module cannot be imported or path is invalid
    """
    if path == "":
        raise ImportError("Empty path")
        
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
            
    assert len(parts) > 0
    part0 = parts[0]
    
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
        
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
            
    return obj


def hydra_get_object(path: str) -> Any:
    """
    Look up an entity based on the dotpath.
    Does not perform any type checks on the entity.
    
    Args:
        path: Dotted path to the object
        
    Returns:
        The requested object
        
    Example:
        >>> import torch
        >>> assert hydra_get_object("torch.float32") is torch.float32
    """
    try:
        obj = hydra_locate(path)
        return obj
    except Exception as e:
        log.error(f"Error getting object at {path} : {e}")
        raise e