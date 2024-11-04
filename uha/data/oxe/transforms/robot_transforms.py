"""Robot-specific transform implementations."""

from dataclasses import dataclass
from typing import Dict, Any, Optional

import tensorflow as tf

from .base import BaseTransform, TransformConfig, GripperProcessor


@dataclass
class RobotTransformConfig(TransformConfig):
    """Configuration for robot-specific transforms."""
    gripper_threshold: float = 0.05
    proprio_dim: Optional[int] = None


class FrankaTransform(BaseTransform):
    """Transform for Franka robot data."""
    
    def _process_actions(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        # Process actions and gripper
        trajectory["action"] = tf.concat([
            trajectory["action"][:, :7],
            GripperProcessor.binarize_gripper(
                trajectory["action"][:, -1:],
                close_threshold=self.config.gripper_threshold
            )
        ], axis=-1)
        return trajectory
        
    def _process_images(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        # Process images according to config if needed
        return trajectory


class UR5Transform(BaseTransform):
    """Transform for UR5 robot data."""
    
    def _process_actions(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        # Process actions
        trajectory["action"] = tf.concat([
            trajectory["action"][:, :6],
            GripperProcessor.binarize_gripper(
                trajectory["action"][:, -1:],
                close_threshold=self.config.gripper_threshold
            )
        ], axis=-1)

        # Process proprio if available
        if "robot_state" in trajectory["observation"]:
            trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"]

        return trajectory


class BimanualTransform(BaseTransform):
    """Transform for bimanual robot data."""
    
    def _process_actions(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        # Process actions for both arms
        left_arm = trajectory["action"][:, :7]
        right_arm = trajectory["action"][:, 7:14]
        
        trajectory["action"] = tf.concat([
            left_arm[:, :-1],
            GripperProcessor.binarize_gripper(
                left_arm[:, -1:],
                close_threshold=self.config.gripper_threshold
            ),
            right_arm[:, :-1],
            GripperProcessor.binarize_gripper(
                right_arm[:, -1:],
                close_threshold=self.config.gripper_threshold
            )
        ], axis=-1)

        # Process proprio if available
        if "joint_states" in trajectory["observation"]:
            trajectory["observation"]["proprio"] = trajectory["observation"]["joint_states"]

        return trajectory