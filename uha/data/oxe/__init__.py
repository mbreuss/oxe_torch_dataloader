import copy
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple, Union, Optional
from dataclasses import dataclass
from pathlib import Path

from uha.data.oxe.oxe_dataset_configs import (
    ActionEncoding, 
    DATASET_REGISTRY,
    OXE_DATASET_CONFIGS,
    ImageConfig
)
from uha.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from uha.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from uha.data.utils.data_utils import NormalizationType
from uha.data.utils.spec import ModuleSpec

logger = logging.getLogger(__name__)

@dataclass
class ImageObsConfig:
    keys: Dict[str, str]
    depth_keys: Dict[str, str]

class DatasetConfigError(Exception):
    """Custom exception for dataset configuration errors"""
    pass


def make_oxe_dataset_kwargs_and_weights(
    data_mix: Union[str, Sequence[Tuple[str, float]]],
    data_dir: str,
    load_camera_views: Sequence[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = False,
    load_language: bool = True,
    dataset_size_limit: int = None,
    force_recompute_dataset_statistics: bool = False,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Generates dataset kwargs and sampling weights for a given dataset mix from the Open X-Embodiment dataset.
    
    Args:
        data_mix: Dataset mix name or sequence of (name, weight) tuples
        data_dir: Base directory for datasets
        load_camera_views: Which camera views to load
        load_depth: Whether to load depth images
        load_proprio: Whether to load proprioception data
        load_language: Whether to load language annotations
        dataset_size_limit: Optional limit on dataset size
        force_recompute_dataset_statistics: Whether to force recompute statistics
        action_proprio_normalization_type: Type of normalization to apply
        
    Returns:
        Tuple of (list of dataset kwargs, list of sampling weights)
    """
    try:
        # Handle named mix case
        data_mix = OXE_NAMED_MIXES[data_mix] if isinstance(data_mix, str) else data_mix

        # Filter out duplicates while preserving order
        filtered_data_mix = list({name: weight for name, weight in data_mix}.items())

        data_kwargs_list = []
        weights = []

        for name, weight in filtered_data_mix:
            try:
                # Generate dataset kwargs for each entry
                dataset_kwargs = make_oxe_dataset_kwargs(
                    name=name,
                    data_dir=data_dir,
                    load_camera_views=load_camera_views,
                    load_depth=load_depth,
                    load_proprio=load_proprio,
                    load_language=load_language,
                    dataset_size_limit=dataset_size_limit,
                    force_recompute_dataset_statistics=force_recompute_dataset_statistics,
                    action_proprio_normalization_type=action_proprio_normalization_type,
                )
                
                # Extract transform config from dataset config
                dataset_config = OXE_DATASET_CONFIGS[name]
                transform_config = {
                    "robot_name": dataset_config.robot_name,
                    "action_space": dataset_config.action_space,
                    "image_obs_keys": convert_image_config_to_dict(dataset_config.image_obs_keys),
                    "depth_obs_keys": convert_image_config_to_dict(dataset_config.depth_obs_keys) if hasattr(dataset_config, 'depth_obs_keys') else None,
                    "proprio_encoding": dataset_config.proprio_encoding,
                    "action_encoding": dataset_config.action_encoding
                }
                
                # Update standardize_fn with config
                dataset_kwargs["standardize_fn"] = ModuleSpec.create(
                    OXE_STANDARDIZATION_TRANSFORMS[name],
                    config=transform_config
                )
                
                data_kwargs_list.append(dataset_kwargs)
                weights.append(weight)

            except ValueError as e:
                logger.warning(f"Skipping dataset '{name}' due to error: {e}")

        if not data_kwargs_list:
            raise ValueError("No valid datasets found in mix")

        # Log dataset mix information
        logger.info("\n" + "=" * 80)
        logger.info(f"Loading {len(data_kwargs_list)} datasets with weights:")
        for kwargs, weight in zip(data_kwargs_list, weights):
            logger.info(f"  {kwargs['name']}: {weight:.3f}")
        logger.info("=" * 80 + "\n")

        return data_kwargs_list, weights

    except Exception as e:
        logger.error(f"Error creating dataset mix: {str(e)}")
        raise


def make_oxe_dataset_kwargs(
    name: str,
    data_dir: str,
    load_camera_views: Sequence[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    dataset_size_limit: int = None,
    force_recompute_dataset_statistics: bool = False,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> Dict[str, Any]:
    """Generates dataset kwargs for a given dataset from Open X-Embodiment."""
    try:
        # Get dataset config from registry
        dataset_config = DATASET_REGISTRY.get_config(name)
        
        # Convert to dictionary and create base kwargs
        config_dict = {
            "name": name,
            "data_dir": dataset_config.data_dir if hasattr(dataset_config, "data_dir") else data_dir,
            "dataset_size_limit": dataset_size_limit,
            "force_recompute_dataset_statistics": force_recompute_dataset_statistics,
            "action_proprio_normalization_type": action_proprio_normalization_type,
            "image_obs_keys": convert_image_config_to_dict(dataset_config.image_obs_keys),
            "depth_obs_keys": convert_image_config_to_dict(dataset_config.depth_obs_keys) if hasattr(dataset_config, 'depth_obs_keys') else None,
            "robot_name": dataset_config.robot_name,
            "action_space": dataset_config.action_space,
            "proprio_encoding": dataset_config.proprio_encoding,
            "action_encoding": dataset_config.action_encoding,
            "transform_type": getattr(dataset_config, "transform_type", "franka"),
            "num_arms": getattr(dataset_config, "num_arms", 1)
        }

        # Create transform config
        transform_config = {
            "robot_name": dataset_config.robot_name,
            "action_space": dataset_config.action_space,
            "image_obs_keys": convert_image_config_to_dict(dataset_config.image_obs_keys),
            "depth_obs_keys": convert_image_config_to_dict(dataset_config.depth_obs_keys) if hasattr(dataset_config, 'depth_obs_keys') else None,
            "proprio_encoding": dataset_config.proprio_encoding,
            "action_encoding": dataset_config.action_encoding
        }

        # Add other configurations
        if load_proprio:
            config_dict["proprio_obs_key"] = "proprio"
        if load_language and not hasattr(dataset_config, "language_key"):
            config_dict["language_key"] = "language_instruction"

        # Add standardize_fn with config
        config_dict["standardize_fn"] = ModuleSpec.create(
            OXE_STANDARDIZATION_TRANSFORMS[name],
            kwargs={"config": transform_config}
        )

        return config_dict

    except Exception as e:
        logger.error(f"Error creating dataset kwargs for {name}: {str(e)}")
        raise

def _normalize_data_dir(data_dir: Union[str, Path]) -> Path:
    """Normalize data directory path."""
    if isinstance(data_dir, str) and data_dir.startswith("~"):
        return Path(os.path.expanduser("~")) / data_dir[1:]
    return Path(data_dir)


def _validate_camera_views(
    name: str,
    load_camera_views: Sequence[str],
    image_config: ImageConfig
) -> None:
    """
    Validates requested camera views against available views in ImageConfig.
    
    Args:
        name: Dataset name
        load_camera_views: Requested camera views to load (e.g., ["primary"])
        image_config: ImageConfig object containing view mappings
    """
    # Get available views (ones that have a non-None value)
    available_views = {
        view: getattr(image_config, view)
        for view in ['primary', 'secondary', 'wrist']
        if getattr(image_config, view) is not None
    }
    
    missing_views = set(load_camera_views) - set(available_views.keys())
    if missing_views:
        available = ", ".join(available_views.keys())
        raise ValueError(
            f"Cannot load {name} with views {missing_views} since they are not available. "
            f"Available views: {available}"
        )

def _configure_optional_features(
    dataset_kwargs: Dict[str, Any],
    load_depth: bool,
    load_proprio: bool,
    load_language: bool,
) -> None:
    """Configures optional dataset features based on flags."""
    if not load_depth and "depth_obs_keys" in dataset_kwargs:
        dataset_kwargs.pop("depth_obs_keys")
    
    if load_proprio:
        dataset_kwargs["proprio_obs_key"] = "proprio"
        
    if load_language and "language_key" not in dataset_kwargs:
        dataset_kwargs["language_key"] = "language_instruction"
    elif not load_language and "language_key" in dataset_kwargs:
        del dataset_kwargs["language_key"]

def _normalize_data_dir(data_dir: Union[str, Path]) -> Path:
    """Normalizes data directory path."""
    data_dir = str(data_dir)
    if data_dir.startswith("~"):
        data_dir = os.path.expanduser("~") + data_dir[1:]
    return Path(data_dir)


def _get_action_normalization_mask(action_encoding: str) -> List[bool]:
    """Get action normalization mask based on action encoding type.
    
    Args:
        action_encoding: Type of action encoding
        
    Returns:
        List of boolean masks for action normalization
    """
    # Define normalization masks for different action types
    action_encoding_masks = {
        "eef_pos": [True] * 6 + [False],  # XYZ + RPY + gripper
        "joint_pos": [True] * 7 + [False],  # 7 joints + gripper
        "joint_pos_bimanual": [True] * 6 + [False] + [True] * 6 + [False],  # 2x(6 joints + gripper)
        "nav_2d": [True] * 2,  # 2D waypoint
        "joint_pos_bimanual_nav": [True] * 6 + [False] + [True] * 6 + [False] + [True] * 2,  # Bimanual + base
    }

    if action_encoding.lower() not in action_encoding_masks:
        raise ValueError(f"Unsupported action encoding: {action_encoding}")
        
    return action_encoding_masks[action_encoding.lower()]

def _validate_camera_views(
    name: str,
    load_camera_views: Sequence[str],
    image_config: Union[ImageConfig, Dict[str, Optional[str]]]
) -> None:
    """Validates requested camera views against available views.
    
    Args:
        name: Dataset name
        load_camera_views: List of camera views to load
        image_config: Image configuration with view mappings
        
    Raises:
        ValueError: If requested views are not available
    """
    # Convert ImageConfig to dict if needed
    if not isinstance(image_config, dict):
        image_config = convert_image_config_to_dict(image_config)
        
    # Get available views (ones that have a non-None value)
    available_views = {
        view: key for view, key in image_config.items()
        if key is not None
    }
    
    # Check for missing views
    missing_views = set(load_camera_views) - set(available_views.keys())
    if missing_views:
        available = ", ".join(available_views.keys())
        raise ValueError(
            f"Cannot load {name} with views {missing_views} since they are not available. "
            f"Available views: {available}"
        )

def _filter_camera_views(
    image_config: Union[ImageConfig, Dict[str, Optional[str]]],
    load_camera_views: Sequence[str]
) -> Dict[str, Optional[str]]:
    """Filter image configuration to only include requested views.
    
    Args:
        image_config: Image configuration with view mappings
        load_camera_views: List of camera views to load
        
    Returns:
        Filtered image configuration dictionary
    """
    # Convert ImageConfig to dict if needed
    if not isinstance(image_config, dict):
        image_config = convert_image_config_to_dict(image_config)
        
    # Create new config with only requested views
    filtered_config = {}
    for view in ["primary", "secondary", "wrist"]:
        if view in load_camera_views and view in image_config:
            filtered_config[view] = image_config[view]
        else:
            filtered_config[view] = None
            
    return filtered_config

def convert_image_config_to_dict(
    image_config: Union[ImageConfig, Dict[str, Optional[str]]]
) -> Dict[str, Optional[str]]:
    """Convert ImageConfig to dictionary format.
    
    Args:
        image_config: ImageConfig object or dictionary
        
    Returns:
        Dictionary of view mappings
    """
    if isinstance(image_config, dict):
        return image_config
        
    return {
        "primary": image_config.primary,
        "secondary": image_config.secondary,
        "wrist": image_config.wrist
    }