import copy
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple, Union

from uha.data.oxe.oxe_dataset_configs import ActionEncoding, OXE_DATASET_CONFIGS
from uha.data.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
from enum import Enum
from uha.data.oxe.oxe_standardization_transforms import (
    OXE_STANDARDIZATION_TRANSFORMS,
    create_action_normalization_mask,
    get_action_space_index,
    NormalizationType
)
# from uha.data.utils.data_utils import NormalizationType
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
    """Generates dataset kwargs for a given dataset from Open X-Embodiment. The returned kwargs can be passed
    directly into `uha.data.dataset.make_dataset_from_rlds`.

    Args:
        name: Name of the dataset to load. See `oxe_dataset_configs.py` for available datasets.
        data_dir: Base data directory that contains the dataset.
        load_camera_views: Which views to load. See `oxe_dataset_configs.py` for available views.
        load_depth: If True, loads corresponding depth channels for each RGB channel.
        load_proprio: If True, loads proprioceptive information.
        load_language: If True, loads language instructions.
        force_recompute_dataset_statistics: If True, recompute dataset statistics.
        action_proprio_normalization_type: Normalization type to use for proprioceptive actions.
    """
    dataset_kwargs = copy.deepcopy(OXE_DATASET_CONFIGS[name])
    robot_type = dataset_kwargs["action_encoding"].name
    control_mode = dataset_kwargs.get("control_mode", "position")
    num_arms = dataset_kwargs.get("num_arms", 1)

    # Create action normalization mask for unified action space
    dataset_kwargs["action_normalization_mask"] = create_action_normalization_mask(robot_type, control_mode)

    # Set action space index
    dataset_kwargs["action_space_index"] = get_action_space_index(robot_type, num_arms, control_mode)

    # adjust loaded camera views
    if missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"])):
        raise ValueError(
            f"Cannot load {name} with views {missing_keys} since they are not available."
        )
    dataset_kwargs["image_obs_keys"] = {
        k: v
        for k, v in dataset_kwargs["image_obs_keys"].items()
        if k in load_camera_views
    }
    dataset_kwargs["depth_obs_keys"] = {
        k: v
        for k, v in dataset_kwargs["depth_obs_keys"].items()
        if k in load_camera_views
    }

    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if load_proprio:
        dataset_kwargs["proprio_obs_key"] = "proprio"
    if load_language and "language_key" not in dataset_kwargs:
        dataset_kwargs["language_key"] = "language_instruction"
    elif not load_language and "language_key" in dataset_kwargs:
        del dataset_kwargs["language_key"]

    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    del dataset_kwargs["proprio_encoding"]
    del dataset_kwargs["action_encoding"]

    dataset_kwargs["standardize_fn"] = ModuleSpec.create(
        OXE_STANDARDIZATION_TRANSFORMS[name]
    )

    if "data_dir" in dataset_kwargs:
        if dataset_kwargs["data_dir"][0] == "~":
            dataset_kwargs["data_dir"] = os.path.expanduser("~") + dataset_kwargs["data_dir"][1:]
        data_dir = dataset_kwargs["data_dir"]
        del dataset_kwargs["data_dir"]

    if dataset_size_limit is not None and dataset_size_limit not in dataset_kwargs:
        dataset_kwargs["dataset_size_limit"] = dataset_size_limit

    if force_recompute_dataset_statistics:
        dataset_kwargs["force_recompute_dataset_statistics"] = True

    # Add information about unified action space
    dataset_kwargs["use_unified_action"] = True
    dataset_kwargs["unified_action_dim"] = 76  # Dimension of the unified action space

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
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    and weights can be passed directly into `octo.data.dataset.make_interleaved_dataset`.

    Args:
        data_mix: List of (dataset name, sampling weight) tuples, or a string specifying a pre-defined mix to
            load from `OXE_NAMED_MIXES`.
        data_dir: Base data directory that contains the datasets.
        load_camera_views: Which views to load. See `oxe_dataset_configs.py` for available views.
        load_depth: If True, loads corresponding depth channels for each RGB channel.
        load_proprio: If True, loads proprioceptive information.
        load_language: If True, loads language instructions.
        force_recompute_dataset_statistics: If True, recompute dataset statistics.
        action_proprio_normalization_type: Normalization type to use for proprioceptive actions.
    Returns:
        Tuple of (dataset_kwargs_list, sampling weights).
    """
    if isinstance(data_mix, str):
        data_mix = OXE_NAMED_MIXES[data_mix]

    filtered_datasets, included_dataset_names = [], []
    for name, weight in data_mix:
        if name not in included_dataset_names:
            filtered_datasets.append((name, weight))
            included_dataset_names.append(name)
        else:
            logging.warning(f"Skipping duplicate: {(name, weight)}.")
    data_mix = filtered_datasets

    data_kwargs_list, weights = [], []
    for name, weight in data_mix:
        try:
            data_kwargs_list.append(
                make_oxe_dataset_kwargs(
                    name,
                    data_dir,
                    load_camera_views,
                    load_depth,
                    load_proprio,
                    load_language,
                    dataset_size_limit,
                    force_recompute_dataset_statistics,
                    action_proprio_normalization_type,
                )
            )
            weights.append(weight)
        except ValueError as e:
            logging.warning(f"Skipping {name} due to error: {e}")

    return data_kwargs_list, weights