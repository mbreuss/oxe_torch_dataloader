"""Base classes for transforms."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import logging

import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass
class ImageKeyConfig:
    """Configuration for image keys."""
    primary: Optional[str] = None
    secondary: Optional[str] = None
    wrist: Optional[str] = None


@dataclass
class TransformConfig:
    """Configuration for transforms."""
    robot_name: str
    action_space: str
    num_arms: int = 1
    image_keys: ImageKeyConfig = ImageKeyConfig()
    depth_keys: Optional[ImageKeyConfig] = None
    gripper_threshold: float = 0.05


class BaseTransform:
    """Base class for trajectory transformations."""
    
    def __init__(self, config: Union[TransformConfig, Dict[str, Any]], *args, **kwargs):
        # Convert dict config to TransformConfig if needed
        if isinstance(config, dict):
            self.config = TransformConfig(
                robot_name=config.get('robot_name', ''),
                action_space=config.get('action_space', ''),
                num_arms=config.get('num_arms', 1),
                image_keys=ImageKeyConfig(**config.get('image_keys', {})),
                depth_keys=config.get('depth_keys'),
                gripper_threshold=config.get('gripper_threshold', 0.05)
            )
        else:
            self.config = config
        self._validate_config()

    def _validate_config(self):
        """Validate transform configuration."""
        if not self.config.robot_name:
            raise ValueError("Robot name must be specified")
        if not self.config.action_space:
            raise ValueError("Action space must be specified")
        if self.config.num_arms < 1:
            raise ValueError("Number of arms must be at least 1")

    def __call__(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Transform trajectory with proper error handling."""
        try:
            self._validate_trajectory(trajectory)
            processed = self._process_trajectory(trajectory)
            return processed
        except Exception as e:
            logger.error(f"Error in transform: {str(e)}")
            raise

    def _process_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Process trajectory data. Override in subclasses."""
        return trajectory

    def _validate_trajectory(self, trajectory: Dict[str, Any]):
        """Validate trajectory structure."""
        required_keys = {"action", "observation"}
        missing_keys = required_keys - set(trajectory.keys())
        if missing_keys:
            raise ValueError(f"Missing required keys in trajectory: {missing_keys}")

    def _process_images(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Process images in trajectory."""
        return trajectory

    def _process_actions(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Process actions in trajectory."""
        return trajectory

    def _add_robot_info(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Add robot information to trajectory."""
        arm_text = "arms" if self.config.num_arms > 1 else "arm"
        trajectory["observation"]["robot_information"] = (
            f"A {self.config.robot_name} robot with {self.config.num_arms} "
            f"{arm_text} controlled by {self.config.action_space} actions"
        )
        return trajectory


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