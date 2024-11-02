"""
Goal relabeling functionality for behavior cloning use cases.
Handles various strategies for sampling and relabeling goals in trajectories.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple, List
import logging

import tensorflow as tf

from uha.data.utils.data_utils import tree_merge

logger = logging.getLogger(__name__)


@dataclass
class RelabelingConfig:
    """Configuration for goal relabeling."""
    min_distance: Optional[int] = None
    max_distance: Optional[int] = None
    frame_diff: int = 1
    future_window: int = 2


class GoalSampler:
    """Handles goal sampling operations."""
    
    @staticmethod
    def sample_uniform_index(traj_len: int, 
                           current_idx: int,
                           min_dist: Optional[int] = None,
                           max_dist: Optional[int] = None) -> tf.Tensor:
        """Sample goal index uniformly from valid range."""
        min_idx = current_idx + (min_dist if min_dist is not None else 0)
        max_idx = tf.minimum(
            current_idx + (max_dist if max_dist is not None else traj_len), 
            traj_len
        )
        
        rand = tf.random.uniform([])
        sampled_idx = tf.cast(
            rand * tf.cast(max_idx - min_idx, tf.float32) + tf.cast(min_idx, tf.float32),
            tf.int32
        )
        
        return tf.minimum(sampled_idx, traj_len - 1)


class FutureFrameManager:
    """Handles future frame operations."""
    
    @staticmethod
    def get_future_indices(traj_len: int, frame_diff: int, window_size: int) -> tf.Tensor:
        """Generate indices for future frames."""
        base_indices = tf.range(traj_len)[:, None]
        future_offsets = tf.range(frame_diff, frame_diff * window_size + 1, frame_diff)
        indices = base_indices + future_offsets
        return tf.minimum(indices, traj_len - 1)


class GoalRelabeler:
    """Base class for goal relabeling strategies."""
    
    def __init__(self, config: RelabelingConfig):
        self.config = config
        self.goal_sampler = GoalSampler()
        self.future_frame_manager = FutureFrameManager()

    def _prepare_goal_observation(self, 
                                obs: Dict[str, Any],
                                indices: tf.Tensor) -> Dict[str, Any]:
        """Prepare goal observation from sampled indices."""
        return {
            "image_primary": tf.gather(obs["image_primary"], indices),
            "pad_mask_dict": tf.nest.map_structure(
                lambda x: tf.gather(x, indices), 
                obs["pad_mask_dict"]
            ),
            "timestep": tf.gather(obs["timestep"], indices)
        }

    def _prepare_future_observation(self, 
                                  obs: Dict[str, Any],
                                  indices: tf.Tensor) -> Dict[str, Any]:
        """Prepare future observation frames."""
        return {"image_primary": tf.gather(obs["image_primary"], indices)}


class UniformGoalRelabeler(GoalRelabeler):
    """Uniform sampling strategy for goal relabeling."""
    
    def __call__(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Relabel goals using uniform sampling."""
        traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
        
        # Sample goal indices
        goal_indices = tf.map_fn(
            lambda i: self.goal_sampler.sample_uniform_index(
                traj_len, i,
                self.config.min_distance,
                self.config.max_distance
            ),
            tf.range(traj_len)
        )
        
        # Prepare goal observation
        goal = self._prepare_goal_observation(traj["observation"], goal_indices)
        traj["task"] = tree_merge(traj["task"], goal)
        
        return traj


class UniformFutureGoalRelabeler(GoalRelabeler):
    """Uniform sampling with future frame tracking."""
    
    def __call__(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Relabel goals and track future frames."""
        traj_len = tf.shape(tf.nest.flatten(traj["observation"])[0])[0]
        
        # Sample goal indices
        goal_indices = tf.map_fn(
            lambda i: self.goal_sampler.sample_uniform_index(
                traj_len, i,
                self.config.min_distance,
                self.config.max_distance
            ),
            tf.range(traj_len)
        )
        
        # Prepare goal observation
        goal = self._prepare_goal_observation(traj["observation"], goal_indices)
        traj["task"] = tree_merge(traj["task"], goal)
        
        # Get future frame indices
        future_indices = self.future_frame_manager.get_future_indices(
            traj_len,
            self.config.frame_diff,
            self.config.future_window
        )
        
        # Add future observations
        traj["future_obs"] = self._prepare_future_observation(
            traj["observation"],
            future_indices
        )
        
        return traj


def create_relabeler(strategy: str, **config_kwargs) -> GoalRelabeler:
    """Create goal relabeler instance."""
    strategies = {
        'uniform': UniformGoalRelabeler,
        'uniform_future': UniformFutureGoalRelabeler
    }
    
    if strategy not in strategies:
        raise ValueError(f"Unknown relabeling strategy: {strategy}")
        
    config = RelabelingConfig(**config_kwargs)
    return strategies[strategy](config)


# Convenience functions for direct use
def uniform(traj: Dict[str, Any], 
           max_goal_distance: Optional[int] = None) -> Dict[str, Any]:
    """Relabel with uniform distribution over future states."""
    relabeler = create_relabeler('uniform', max_distance=max_goal_distance)
    return relabeler(traj)


def uniform_and_future(traj: Dict[str, Any],
                      max_goal_distance: Optional[int] = None,
                      frame_diff: int = 1) -> Dict[str, Any]:
    """Relabel with uniform distribution and track future frames."""
    relabeler = create_relabeler(
        'uniform_future',
        max_distance=max_goal_distance,
        frame_diff=frame_diff
    )
    return relabeler(traj)


def bound_uniform(traj: Dict[str, Any],
                 min_bound: Optional[int] = None,
                 max_bound: Optional[int] = None) -> Dict[str, Any]:
    """Relabel with uniform distribution within bounds."""
    relabeler = create_relabeler(
        'uniform',
        min_distance=min_bound,
        max_distance=max_bound
    )
    return relabeler(traj)


def bound_uniform_and_future(traj: Dict[str, Any],
                           min_bound: Optional[int] = None,
                           max_bound: Optional[int] = None,
                           frame_diff: int = 1) -> Dict[str, Any]:
    """Relabel with bounded uniform distribution and track future frames."""
    relabeler = create_relabeler(
        'uniform_future',
        min_distance=min_bound,
        max_distance=max_bound,
        frame_diff=frame_diff
    )
    return relabeler(traj)