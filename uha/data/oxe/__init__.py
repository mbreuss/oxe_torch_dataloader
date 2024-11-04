import copy
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple, Union, Mapping
from dataclasses import dataclass
from pathlib import Path

from uha.data.oxe.oxe_dataset_configs import ActionEncoding, OXE_DATASET_CONFIGS
from uha.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from uha.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from uha.data.utils.data_utils import NormalizationType
from uha.data.utils.spec import ModuleSpec

from uha.data.oxe.oxe_dataset_configs import ImageConfig


@dataclass
class ImageObsConfig:
    keys: Dict[str, str]
    depth_keys: Dict[str, str]

class DatasetConfigError(Exception):
    """Custom exception for dataset configuration errors"""
    pass


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
    """
    Generates dataset kwargs for a given dataset from Open X-Embodiment.
    """
    # Deep copy to avoid mutating the original config
    dataset_kwargs = copy.deepcopy(OXE_DATASET_CONFIGS[name])

    # Set action normalization masks
    action_encoding_masks = {
        ActionEncoding.EEF_POS: [True] * 6 + [False],
        ActionEncoding.JOINT_POS: [True] * 7 + [False],
        ActionEncoding.JOINT_POS_BIMANUAL: [True] * 6 + [False] + [True] * 6 + [False],
        ActionEncoding.NAV_2D: [True] * 2,
        ActionEncoding.JOINT_POS_BIMANUAL_NAV: [True] * 6 + [False] + [True] * 6 + [False] + [True] * 2,
    }

    if dataset_kwargs.action_encoding in action_encoding_masks:
        dataset_kwargs.action_normalization_mask = action_encoding_masks[dataset_kwargs.action_encoding]
    else:
        raise ValueError(f"Unsupported action encoding: {dataset_kwargs.action_encoding}")

    # Handle camera views
    missing_views = set(load_camera_views) - set(view for view in ['primary', 'secondary', 'wrist']
                                                if getattr(dataset_kwargs.image_obs_keys, view) is not None)
    if missing_views:
        raise ValueError(f"Cannot load {name} with views {missing_views} since they are not available.")
    
    # Filter image observation keys based on requested views
    dataset_kwargs.image_obs_keys = ImageConfig(
        primary=dataset_kwargs.image_obs_keys.primary if 'primary' in load_camera_views else None,
        secondary=dataset_kwargs.image_obs_keys.secondary if 'secondary' in load_camera_views else None,
        wrist=dataset_kwargs.image_obs_keys.wrist if 'wrist' in load_camera_views else None
    )

    # Filter depth observation keys if they exist and depth is requested
    if hasattr(dataset_kwargs, 'depth_obs_keys'):
        dataset_kwargs.depth_obs_keys = ImageConfig(
            primary=dataset_kwargs.depth_obs_keys.primary if 'primary' in load_camera_views else None,
            secondary=dataset_kwargs.depth_obs_keys.secondary if 'secondary' in load_camera_views else None,
            wrist=dataset_kwargs.depth_obs_keys.wrist if 'wrist' in load_camera_views else None
        )

    # Handle optional features
    if not load_depth and hasattr(dataset_kwargs, "depth_obs_keys"):
        delattr(dataset_kwargs, "depth_obs_keys")
    if load_proprio:
        dataset_kwargs.proprio_obs_key = "proprio"
    if load_language and not hasattr(dataset_kwargs, "language_key"):
        dataset_kwargs.language_key = "language_instruction"
    elif not load_language and hasattr(dataset_kwargs, "language_key"):
        delattr(dataset_kwargs, "language_key")

    # Set standardization and normalization
    dataset_kwargs.action_proprio_normalization_type = action_proprio_normalization_type
    dataset_kwargs.standardize_fn = ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS[name])

    # Handle data directory
    if hasattr(dataset_kwargs, "data_dir"):
        path_str = str(dataset_kwargs.data_dir)
        if path_str.startswith("~"):
            dataset_kwargs.data_dir = os.path.expanduser("~") + path_str[1:]
        data_dir = dataset_kwargs.data_dir
        delattr(dataset_kwargs, "data_dir")

    # Handle size limits and statistics
    if dataset_size_limit is not None:
        dataset_kwargs.dataset_size_limit = dataset_size_limit
    if force_recompute_dataset_statistics:
        dataset_kwargs.force_recompute_dataset_statistics = True

    # Remove encoding information
    delattr(dataset_kwargs, "proprio_encoding")
    delattr(dataset_kwargs, "action_encoding")

    return {"name": name, "data_dir": str(data_dir), **dataset_kwargs.__dict__}


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
    """

    # Handle named mix case
    data_mix = OXE_NAMED_MIXES[data_mix] if isinstance(data_mix, str) else data_mix

    # Filter out duplicates while preserving the order
    filtered_data_mix = {name: weight for name, weight in data_mix}.items()

    data_kwargs_list, weights = [], []
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
            data_kwargs_list.append(dataset_kwargs)
            weights.append(weight)

        except ValueError as e:
            logging.warning(f"Skipping dataset '{name}' due to error: {e}")

    return data_kwargs_list, weights


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