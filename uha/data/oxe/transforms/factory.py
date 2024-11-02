from typing import Dict, Type

from .base import BaseTransform, TransformConfig
from .robot_transforms import (
    FrankaTransform,
    UR5Transform,
    BimanualTransform,
    RobotTransformConfig
)


class TransformFactory:
    """Factory for creating trajectory transforms."""
    
    _transforms: Dict[str, Type[BaseTransform]] = {
        'franka': FrankaTransform,
        'ur5': UR5Transform,
        'bimanual': BimanualTransform
    }
    
    _configs: Dict[str, Type[TransformConfig]] = {
        'franka': RobotTransformConfig,
        'ur5': RobotTransformConfig,
        'bimanual': RobotTransformConfig
    }
    
    @classmethod
    def create(cls, transform_type: str, **config_kwargs) -> BaseTransform:
        """Create transform instance."""
        if transform_type not in cls._transforms:
            raise ValueError(f"Unknown transform type: {transform_type}")
            
        transform_cls = cls._transforms[transform_type]
        config_cls = cls._configs[transform_type]
        
        config = config_cls(**config_kwargs)
        return transform_cls(config)