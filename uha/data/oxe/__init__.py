import copy
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

from uha.data.oxe.oxe_dataset_configs import ActionEncoding, OXE_DATASET_CONFIGS
from uha.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from uha.data.oxe.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from uha.data.utils.data_utils import NormalizationType
from uha.data.utils.spec import ModuleSpec


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
    Generates dataset kwargs for a given dataset from Open X-Embodiment. The returned kwargs can be passed
    directly into `uha.data.dataset.make_dataset_from_rlds`.
    """
    # Deep copy to avoid mutating the original config
    dataset_kwargs = copy.deepcopy(OXE_DATASET_CONFIGS[name])

    # Set action normalization masks based on encoding type
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

    # Adjust loaded camera views
    # Extract non-None keys from ImageConfig object
    image_obs_keys = {k: v for k, v in vars(dataset_kwargs.image_obs_keys).items() if v is not None}

    # Now calculate missing keys
    missing_keys = set(load_camera_views) - set(image_obs_keys.keys())

    if missing_keys:
        raise ValueError(f"Unavailable views: {missing_keys}")

    dataset_kwargs.image_obs_keys = {k: v for k, v in dataset_kwargs.image_obs_keys.items() if k in load_camera_views}
    dataset_kwargs.depth_obs_keys = {k: v for k, v in dataset_kwargs.depth_obs_keys.items() if k in load_camera_views}

    # Modify depth and language keys based on flags
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys", None)
    dataset_kwargs["proprio_obs_key"] = "proprio" if load_proprio else None
    if load_language:
        dataset_kwargs.setdefault("language_key", "language_instruction")
    else:
        dataset_kwargs.pop("language_key", None)

    # Set additional fields
    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type
    dataset_kwargs["standardize_fn"] = ModuleSpec.create(OXE_STANDARDIZATION_TRANSFORMS[name])

    # Handle data directory and size limits
    if "data_dir" in dataset_kwargs:
        dataset_kwargs["data_dir"] = os.path.expanduser(dataset_kwargs["data_dir"])
    if dataset_size_limit is not None:
        dataset_kwargs["dataset_size_limit"] = dataset_size_limit
    if force_recompute_dataset_statistics:
        dataset_kwargs["force_recompute_dataset_statistics"] = True

    return {"name": name, "data_dir": data_dir, **dataset_kwargs}


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
