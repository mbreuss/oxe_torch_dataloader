"""Transform factory module."""

from typing import Dict, Any
import importlib

from .base import BaseTransform, TransformConfig
from .robot_transforms import RobotTransformConfig


class TransformFactory:
    """Factory for creating trajectory transforms."""
    
    _configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_config(cls, transform_path: str, config: Dict[str, Any]):
        """Register configuration for a transform path."""
        cls._configs[transform_path] = config
    
    @classmethod
    def create(cls, transform_path: str, **config_kwargs) -> BaseTransform:
        """
        Create transform instance from module path.
        
        Args:
            transform_path: Module path in format 'module.submodule:class'
            config_kwargs: Additional configuration arguments (overrides registered config)
        """
        try:
            module_path, class_name = transform_path.split(':')
            module = importlib.import_module(module_path)
            transform_cls = getattr(module, class_name)
            
            # Merge registered config with additional kwargs
            base_config = cls._configs.get(transform_path, {})
            config_dict = {**base_config, **config_kwargs}
            
            config = RobotTransformConfig(**config_dict)
            return transform_cls(config)
        except Exception as e:
            raise ImportError(f"Error creating transform from path {transform_path}: {str(e)}")

    @classmethod
    def create_from_name(cls, transform_type: str, **config_kwargs) -> BaseTransform:
        """Create transform instance from predefined type."""
        transform_path = f"uha.data.oxe.transforms.robot_transforms:{transform_type.capitalize()}Transform"
        return cls.create(transform_path, **config_kwargs)