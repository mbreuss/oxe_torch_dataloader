"""Transform factory module."""

from typing import Dict, Any, Optional
import importlib

from .base import BaseTransform, TransformConfig
from .robot_transforms import RobotTransformConfig
from .robot_transforms import FrankaTransform, UR5Transform, BimanualTransform
# import ImageKeyConfig
from .base import ImageKeyConfig
import logging

logger = logging.getLogger(__name__)


class TransformFactory:
    """Factory for creating trajectory transforms."""
    
    # Initialize configs dictionary as class variable
    _configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def create(cls, transform_path: str, **config_kwargs) -> BaseTransform:
        """Create transform instance from module path."""
        try:
            # Get registered config and merge with kwargs
            base_config = cls._configs.get(transform_path, {})
            config_dict = {**base_config, **config_kwargs}
            
            # Split module path
            module_path, class_name = transform_path.split(':')
            module = importlib.import_module(module_path)
            transform_cls = getattr(module, class_name)
            
            # Create config
            config = RobotTransformConfig(
                robot_name=config_dict.get('robot_name', ''),
                action_space=config_dict.get('action_space', ''),
                num_arms=config_dict.get('num_arms', 1),
                image_keys=ImageKeyConfig(**config_dict.get('image_keys', {})),
                depth_keys=config_dict.get('depth_keys'),
                gripper_threshold=config_dict.get('gripper_threshold', 0.05)
            )
            
            transform = transform_cls(config)
            
            # Validate transform
            if not callable(transform):
                raise ValueError(f"Transform {class_name} must be callable")
                
            return transform
            
        except Exception as e:
            logger.error(f"Error creating transform from path {transform_path}: {str(e)}")
            raise

    @classmethod
    def create_from_name(cls, transform_type: str, **config_kwargs) -> BaseTransform:
        """Create transform instance from predefined type."""
        transform_path = f"uha.data.oxe.transforms.robot_transforms:{transform_type.capitalize()}Transform"
        return cls.create(transform_path, **config_kwargs)

    @classmethod
    def get_config(cls, transform_path: str) -> Optional[Dict[str, Any]]:
        """Get registered configuration for transform path."""
        return cls._configs.get(transform_path)

    @classmethod
    def clear_configs(cls) -> None:
        """Clear all registered configurations."""
        cls._configs.clear()
    
    @classmethod
    def register_config(cls, transform_path: str, config: Dict[str, Any]):
        """Register configuration for a transform path."""
        cls._configs[transform_path] = config
