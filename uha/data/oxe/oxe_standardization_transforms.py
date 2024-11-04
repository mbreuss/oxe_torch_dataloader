"""Dataset standardization transforms registry with complete dataset coverage."""

from typing import Dict, Any
import logging

from .transforms.base import ImageKeyConfig
from .configs.dataset_configs import DATASET_CONFIGS
from .transforms.factory import TransformFactory

logger = logging.getLogger(__name__)


def _make_transform_spec(robot_name: str, action_space: str, image_keys: Dict[str, Any], 
                        transform_type: str = "franka", **kwargs) -> str:
    """Create transform specification with config registration."""
    transform_cls = {
        "franka": "FrankaTransform",
        "ur5": "UR5Transform",
        "bimanual": "BimanualTransform"
    }[transform_type]
    
    transform_path = f"uha.data.oxe.transforms.robot_transforms:{transform_cls}"
    
    # Create full config
    config = {
        "robot_name": robot_name,
        "action_space": action_space,
        "image_keys": image_keys,
        **kwargs
    }
    
    # Store config for later
    TransformFactory.register_config(transform_path, config)
    
    return transform_path


def generate_transform_specs() -> Dict[str, str]:
    """Generate all transform specifications from configurations."""
    return {
        name: _make_transform_spec(**config)
        for name, config in DATASET_CONFIGS.items()
    }


# Generate all transform specifications
OXE_STANDARDIZATION_TRANSFORMS = generate_transform_specs()