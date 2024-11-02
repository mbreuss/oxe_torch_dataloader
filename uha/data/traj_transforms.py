"""
Trajectory transformation implementations for the data pipeline.
These transforms operate on complete trajectories rather than individual frames.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union, List
import logging

import tensorflow as tf
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for trajectory chunking."""
    window_size: int = 1
    action_horizon: int = 1


@dataclass
class PaddingConfig:
    """Configuration for padding operations."""
    max_action_dim: Optional[int] = None
    max_proprio_dim: Optional[int] = None


class PadMaskGenerator:
    """Handles creation and management of padding masks."""

    @staticmethod
    def create_pad_mask_dict(traj: Dict[str, Any]) -> Dict[str, Dict[str, tf.Tensor]]:
        """Create padding mask dictionary for trajectory."""
        traj_len = tf.shape(traj["action"])[0]
        pad_masks = {}
        
        for key in ["observation", "task"]:
            pad_mask_dict = {}
            for subkey in traj[key]:
                if traj[key][subkey].dtype == tf.string:
                    # For language instructions, images, and depth
                    pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
                else:
                    # Other keys aren't padding
                    pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
            pad_masks[key] = pad_mask_dict
            
        return pad_masks


class DimensionPadder:
    """Handles padding operations for actions and proprio."""
    
    @staticmethod
    def pad_actions_and_proprio(
            traj: Dict[str, Any],
            config: PaddingConfig
    ) -> Dict[str, Any]:
        """Pad actions and proprio to specified dimensions."""
        # Initialize action mask
        traj["action_pad_mask"] = tf.ones_like(traj["action"], dtype=tf.bool)
        
        # Pad actions if needed
        if config.max_action_dim is not None:
            action_dim = traj["action"].shape[-1]
            if action_dim > config.max_action_dim:
                raise ValueError(
                    f"action_dim ({action_dim}) > max_action_dim ({config.max_action_dim})"
                )
            
            padding = [[0, 0]] * (len(traj["action"].shape) - 1) + \
                     [[0, config.max_action_dim - action_dim]]
                     
            for key in ["action", "action_pad_mask"]:
                traj[key] = tf.pad(traj[key], padding)

        # Pad proprio if needed
        if config.max_proprio_dim is not None and "proprio" in traj["observation"]:
            proprio_dim = traj["observation"]["proprio"].shape[-1]
            if proprio_dim > config.max_proprio_dim:
                raise ValueError(
                    f"proprio_dim ({proprio_dim}) > max_proprio_dim ({config.max_proprio_dim})"
                )
                
            padding = [[0, 0], [0, config.max_proprio_dim - proprio_dim]]
            traj["observation"]["proprio"] = tf.pad(
                traj["observation"]["proprio"],
                padding
            )

        return traj


class TrajectoryChunker:
    """Handles trajectory chunking operations."""
    
    def __init__(self, config: ChunkConfig):
        self.config = config

    def chunk_trajectory(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk trajectory into windows."""
        traj_len = tf.shape(traj["action"])[0]

        # Create history indices
        history_indices = tf.range(traj_len)[:, None] + tf.range(
            -self.config.window_size + 1, 1
        )
        
        # Create timestep padding mask
        timestep_pad_mask = history_indices >= 0
        
        # Handle boundary conditions
        history_indices = tf.maximum(history_indices, 0)
        
        # Chunk observations
        traj["observation"] = tf.nest.map_structure(
            lambda x: tf.gather(x, history_indices), 
            traj["observation"]
        )
        traj["observation"]["timestep_pad_mask"] = timestep_pad_mask

        # Chunk actions
        if len(traj["action"].shape) == 2:
            action_chunk_indices = tf.range(traj_len)[:, None] + tf.range(
                self.config.action_horizon
            )
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            traj["action"] = tf.gather(traj["action"], action_chunk_indices)
        else:
            if traj["action"].shape[1] < self.config.action_horizon:
                raise ValueError(
                    f"action_horizon ({self.config.action_horizon}) > "
                    f"pre-chunked action dimension ({traj['action'].shape[1]})"
                )
            traj["action"] = traj["action"][:, :self.config.action_horizon]

        # Add history axis to actions
        traj["action"] = tf.gather(traj["action"], history_indices)

        # Handle goal timesteps
        if "timestep" in traj["task"]:
            goal_timestep = traj["task"]["timestep"]
        else:
            goal_timestep = tf.fill([traj_len], traj_len - 1)

        # Compute relative goal timesteps
        t, w, h = tf.meshgrid(
            tf.range(traj_len),
            tf.range(self.config.window_size),
            tf.range(self.config.action_horizon),
            indexing="ij"
        )
        relative_goal_timestep = goal_timestep[:, None, None] - (
            t - (self.config.window_size + 1) + w + h
        )
        
        # Add task completion information
        traj["observation"]["task_completed"] = relative_goal_timestep <= 0

        # Update action padding mask
        traj["action_pad_mask"] = tf.logical_and(
            traj["action_pad_mask"][:, None, None, :]
            if len(traj["action_pad_mask"].shape) == 2
            else traj["action_pad_mask"][:, None, :],
            tf.logical_not(traj["observation"]["task_completed"])[:, :, :, None]
        )

        return traj


class SubsampleGenerator:
    """Handles trajectory subsampling operations."""
    
    @staticmethod
    def subsample_trajectory(traj: Dict[str, Any], subsample_length: int) -> Dict[str, Any]:
        """Subsample trajectory to specified length."""
        traj_len = tf.shape(traj["action"])[0]
        
        if traj_len > subsample_length:
            indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
            return tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
            
        return traj


# Create instances of transform handlers
pad_mask_generator = PadMaskGenerator()
dimension_padder = DimensionPadder()


def chunk_act_obs(
        traj: Dict[str, Any],
        window_size: int = 1,
        action_horizon: int = 1
) -> Dict[str, Any]:
    """Chunk actions and observations in trajectory."""
    config = ChunkConfig(window_size=window_size, action_horizon=action_horizon)
    chunker = TrajectoryChunker(config)
    return chunker.chunk_trajectory(traj)


def subsample(traj: Dict[str, Any], subsample_length: int) -> Dict[str, Any]:
    """Subsample trajectory to specified length."""
    return SubsampleGenerator.subsample_trajectory(traj, subsample_length)


def add_pad_mask_dict(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Add padding mask dictionary to trajectory."""
    pad_masks = pad_mask_generator.create_pad_mask_dict(traj)
    traj["observation"]["pad_mask_dict"] = pad_masks["observation"]
    traj["task"]["pad_mask_dict"] = pad_masks["task"]
    return traj


def pad_actions_and_proprio(
        traj: Dict[str, Any],
        max_action_dim: Optional[int] = None,
        max_proprio_dim: Optional[int] = None
) -> Dict[str, Any]:
    """Pad actions and proprio to specified dimensions."""
    config = PaddingConfig(
        max_action_dim=max_action_dim,
        max_proprio_dim=max_proprio_dim
    )
    return dimension_padder.pad_actions_and_proprio(traj, config)