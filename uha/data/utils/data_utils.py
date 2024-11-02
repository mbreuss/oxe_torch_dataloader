from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import hashlib
import json
import logging
import os

import dlimp as dl
import numpy as np
import tensorflow as tf
import tqdm

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