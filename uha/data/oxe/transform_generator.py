"""Helper to generate all transforms from configurations."""

from typing import Dict

from .dataset_configs import DATASET_CONFIGS
from .transforms.base import ImageKeyConfig


def generate_transform_specs() -> Dict[str, str]:
    """Generate all transform specifications from configurations."""
    
    def _make_transform_spec(config: dict) -> str:
        from .transforms.factory import TransformFactory
        
        transform_type = config.pop("transform_type", "franka")
        transform_cls = {
            "franka": "FrankaTransform",
            "ur5": "UR5Transform",
            "bimanual": "BimanualTransform"
        }[transform_type]
        
        transform_path = f"uha.data.oxe.transforms.robot_transforms:{transform_cls}"
        TransformFactory.register_config(transform_path, config)
        return transform_path
    
    return {
        name: _make_transform_spec(config)
        for name, config in DATASET_CONFIGS.items()
    }

# Generate all transform specs
OXE_STANDARDIZATION_TRANSFORMS = generate_transform_specs()
