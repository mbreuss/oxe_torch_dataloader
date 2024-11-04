"""Registry for dataset standardization transforms."""

from typing import Dict

from .base import TransformConfig
from .robot_transforms import FrankaTransform, UR5Transform, BimanualTransform


class TransformsRegistry:
    """Registry for dataset standardization transforms."""
    
    def __init__(self):
        self._transforms: Dict[str, str] = {}
        self._initialize_transforms()

    def _initialize_transforms(self):
        """Initialize all standardization transforms."""
        # Common configurations
        franka_config = TransformConfig(
            name="Franka",
            control_type="delta end-effector"
        )
        
        ur5_config = TransformConfig(
            name="UR5",
            control_type="delta end-effector"
        )
        
        bimanual_config = TransformConfig(
            name="ViperX",
            control_type="absolute joint",
            num_arms=2
        )

        # Map dataset names to transform classes with configurations
        transforms = {
            # Franka-based datasets
            "libero_spatial_no_noops": (FrankaTransform, franka_config),
            "libero_object_no_noops": (FrankaTransform, franka_config),
            "libero_goal_no_noops": (FrankaTransform, franka_config),
            "libero_10_no_noops": (FrankaTransform, franka_config),
            "kit_irl_real_kitchen_lang": (FrankaTransform, franka_config),
            "kit_irl_real_kitchen_vis": (FrankaTransform, franka_config),
            
            # UR5-based datasets
            "berkeley_autolab_ur5": (UR5Transform, ur5_config),
            
            # Bimanual datasets
            "aloha_mobile": (BimanualTransform, bimanual_config),
            "aloha_static_dataset": (BimanualTransform, bimanual_config),
            "aloha_dagger_dataset": (BimanualTransform, bimanual_config),
        }

        # Convert to ModuleSpec format
        self._transforms = {
            name: f"uha.data.oxe.transforms.robot_transforms:{transform.__name__}"
            for name, (transform, _) in transforms.items()
        }

    def get_transform(self, dataset_name: str) -> str:
        """Get transform path for a dataset."""
        if dataset_name not in self._transforms:
            raise ValueError(f"No transform found for dataset: {dataset_name}")
        return self._transforms[dataset_name]

    def register_transform(self, dataset_name: str, transform_path: str):
        """Register a new transform."""
        self._transforms[dataset_name] = transform_path


# Create global registry instance
TRANSFORMS_REGISTRY = TransformsRegistry()

# Export transforms dictionary
OXE_STANDARDIZATION_TRANSFORMS = {
    name: transform_path
    for name, transform_path in TRANSFORMS_REGISTRY._transforms.items()
}