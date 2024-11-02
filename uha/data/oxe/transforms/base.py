"""Base classes and utilities for transforms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class TransformConfig:
    """Base configuration for transforms."""
    robot_name: str
    action_space: str
    num_arms: int = 1


class BaseTransform(ABC):
    """Base class for trajectory transformations."""
    
    def __init__(self, config: TransformConfig):
        self.config = config

    @abstractmethod
    def __call__(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trajectory."""
        pass

    def _add_robot_information(self) -> str:
        """Generate robot information string."""
        arm_text = "arms" if self.config.num_arms > 1 else "arm"
        return (f"A {self.config.robot_name} robot with {self.config.num_arms} "
                f"{arm_text} controlled by {self.config.action_space} actions")

    def _validate_trajectory(self, trajectory: Dict[str, Any]) -> None:
        """Validate trajectory structure."""
        required_keys = {"action", "observation"}
        missing_keys = required_keys - set(trajectory.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in trajectory: {missing_keys}")


class GripperProcessor:
    """Helper class for gripper-related operations."""
    
    @staticmethod
    def binarize_gripper(actions: tf.Tensor, 
                        open_threshold: float = 0.95,
                        close_threshold: float = 0.05) -> tf.Tensor:
        """Binarize gripper actions."""
        return tf.concat([
            actions[..., :-1],
            tf.cast(actions[..., -1:] > open_threshold, actions.dtype)
        ], axis=-1)

    @staticmethod
    def process_joint_actions(actions: tf.Tensor, 
                            gripper_idx: int = -1) -> tf.Tensor:
        """Process joint actions with gripper."""
        gripper = tf.cast(actions[..., gripper_idx:], actions.dtype)
        return tf.concat([actions[..., :gripper_idx], gripper], axis=-1)