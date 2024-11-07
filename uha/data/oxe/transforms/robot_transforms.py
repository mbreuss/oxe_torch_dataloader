"""Robot-specific transform implementations."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging
import tensorflow as tf

from .base import BaseTransform, TransformConfig, GripperProcessor

logger = logging.getLogger(__name__)

@dataclass
class RobotTransformConfig(TransformConfig):
    """Configuration for robot-specific transforms."""
    gripper_threshold: float = 0.05
    proprio_dim: Optional[int] = None


class FrankaTransform(BaseTransform):
    """Transform for Franka robot data."""
    
    def _process_trajectory(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Process trajectory for Franka robot."""
        try:
            # Create a new dictionary to avoid modifying input
            result = dict(trajectory)
            
            # Process actions
            if "action" in result:
                result["action"] = tf.concat([
                    trajectory["action"][:, :7],
                    GripperProcessor.binarize_gripper(
                        trajectory["action"][:, -1:],
                        close_threshold=self.config.gripper_threshold
                    )
                ], axis=-1)

            # Process observations
            if "observation" in result:
                result["observation"] = dict(trajectory["observation"])
                
            # Add robot information if configured
            if hasattr(self.config, 'add_robot_information') and self.config.add_robot_information:
                result["observation"]["robot_information"] = (
                    f"A {self.config.robot_name} robot with {self.config.num_arms} "
                    f"{'arms' if self.config.num_arms > 1 else 'arm'} controlled by "
                    f"{self.config.action_space} actions"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing trajectory: {str(e)}")
            raise


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