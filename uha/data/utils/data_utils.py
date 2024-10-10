from enum import Enum
from fnmatch import fnmatch
import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import dlimp as dl
import numpy as np
import tensorflow as tf
import tqdm

import logging.config

log = logging.getLogger(__name__)

def hydra_get_object(path: str) -> Any:
  """
  !Copied for compatibility!
  Look up an entity based on the dotpath.
  Does not perform any type checks on the entity.

  >>> import my_module
  >>> from hydra.utils import get_object
  >>> assert get_object("my_module.my_object") is my_module.my_object
  """
  try:
      obj = hydra_locate(path)
      return obj
  except Exception as e:
      log.error(f"Error getting object at {path} : {e}")
      raise e
  
def hydra_locate(path: str) -> Any:
    """
    !Copied for compatibility!
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]
        try:
            obj = getattr(obj, part)
        except AttributeError as exc_attr:
            parent_dotpath = ".".join(parts[:m])
            if isinstance(obj, ModuleType):
                mod = ".".join(parts[: m + 1])
                try:
                    obj = import_module(mod)
                    continue
                except ModuleNotFoundError as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is importable from module '{parent_dotpath}'?"
                    ) from exc_import
                except Exception as exc_import:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                    ) from exc_import
            raise ImportError(
                f"Error loading '{path}':\n{repr(exc_attr)}"
                + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
            ) from exc_attr
    return obj

def tree_map(fn: Callable, tree: dict) -> dict:
    """Maps a function over a nested dictionary."""
    return {
        k: tree_map(fn, v) if isinstance(v, dict) else fn(v) for k, v in tree.items()
    }


def tree_merge(*trees: dict) -> dict:
    """Merges a list of nested dictionaries, with later dictionaries overriding earlier ones."""
    merged = {}
    for tree in trees:
        for k, v in tree.items():
            if isinstance(v, dict):
                merged[k] = tree_merge(merged.get(k, {}), v)
            else:
                merged[k] = v
    return merged


class NormalizationType(str, Enum):
    """Defines supported normalization schemes for action and proprio."""

    NORMAL = "normal"  # normalize to mean 0, std 1
    BOUNDS = "bounds"  # normalize to [-1, 1]


def to_padding(tensor: tf.Tensor) -> tf.Tensor:
    if tf.debugging.is_numeric_tensor(tensor):
        return tf.zeros_like(tensor)
    elif tensor.dtype == tf.string:
        return tf.fill(tf.shape(tensor), "")
    else:
        raise ValueError(f"Cannot generate padding for tensor of type {tensor.dtype}.")


def sample_match_keys_uniform(d: dict, key_template: str):
    """Samples uniformly from all keys fnmatching the template."""
    match_keys = [key for key in d.keys() if fnmatch(key, key_template)]
    if not match_keys:
        raise ValueError(f"No matching key found for {key_template}. Keys: {d.keys()}")
    logging.info(f"Sampling uniformly across keys: {match_keys}")
    if len(match_keys) > 1:
        stacked = tf.stack([d[key] for key in match_keys])
        idx = tf.random.uniform((), 0, len(stacked) - 1, dtype=tf.int32)
        return stacked[idx]
    else:
        return d[match_keys[0]]


def filter_by_language_key(traj, *, language_key_template):
    match_keys = [key for key in traj.keys() if fnmatch(key, language_key_template)]
    if len(match_keys) == 0:
        raise ValueError(f"No matching key found for {language_key_template}. Keys: {traj.keys()}")
    
    labels = tf.stack([traj[key] for key in match_keys], axis=0)
    # if _any_ label in _any_ step is not empty, return True
    return tf.math.reduce_any(labels != "")


def filter_by_task(traj, *, task_templates,negative_task_templates):
    metadata = traj["episode_metadata"]["file_path"]
    pos_match = False
    pos_match = tf.math.reduce_any([task_template in metadata for task_template in task_templates])
    neg_match = False
    neg_match = tf.math.reduce_any([task_template in metadata for task_template in negative_task_templates])
    valid_traj = pos_match and not neg_match
    return valid_traj


def filter_by_task_and_language_key(traj, *, language_key_gt_template,language_key_NILS_template,gt_task_templates,NILS_task_templates,negative_task_templates):
    metadata = traj["traj_metadata"]["episode_metadata"]["file_path"][0]
    logging.info(f"metadata: {metadata}")
    logging.info(traj[language_key_gt_template])
    has_gt_annotation = False
    has_NILS_annotation = False
    if traj[language_key_gt_template][0] != "":
        has_gt_annotation = True
    if traj[language_key_NILS_template][0] != "":
        has_NILS_annotation = True

    if len(gt_task_templates) > 0:
        if gt_task_templates[0] == "REMAINING":
            pos_match_gt = True
        else:
            pos_pattern = '|'.join([".*" + template + ".*" for template in gt_task_templates])
            pos_match_gt = tf.reduce_any(tf.strings.regex_full_match(metadata, pos_pattern))
    else:
        pos_match_gt = False

    if len(NILS_task_templates) > 0:
        pos_pattern_NILS = '|'.join([".*" + template + ".*" for template in NILS_task_templates])
        pos_match_NILS = (tf.strings.regex_full_match(metadata, pos_pattern_NILS))
        logging.info(f"pos_match_NILS: {pos_match_NILS}")
    else:
        pos_match_NILS = False

    if len(negative_task_templates) > 0:
        neg_pattern = '|'.join([".*" + template + ".*" for template in negative_task_templates])
        neg_match = tf.reduce_any(tf.strings.regex_full_match(metadata, neg_pattern))
    else:
        neg_match = False

    valid_traj = ((pos_match_gt and has_gt_annotation) or (pos_match_NILS and has_NILS_annotation)) and not neg_match
    valid_traj = tf.cast(valid_traj, tf.bool)

    logging.info(f"valid_traj: {valid_traj}")

    return valid_traj


def pprint_data_mixture(
    dataset_kwargs_list: List[Dict[str, Any]], dataset_weights: List[int]
) -> None:
    print(
        "\n######################################################################################"
    )
    print(
        f"# Loading the following {len(dataset_kwargs_list)} datasets (incl. sampling weight):{'': >24} #"
    )
    for dataset_kwargs, weight in zip(dataset_kwargs_list, dataset_weights):
        pad = 80 - len(dataset_kwargs["name"])
        print(f"# {dataset_kwargs['name']}: {weight:=>{pad}f} #")
    print(
        "######################################################################################\n"
    )


def get_dataset_statistics(
    dataset: dl.DLataset,
    hash_dependencies: Tuple[str, ...],
    save_dir: Optional[str] = None,
    force_recompute: bool = False,
    unified_action_dim: int = 76
) -> dict:
    """Either computes the statistics of a dataset or loads them from a cache file if this function has been
    called before with the same `hash_dependencies`. Currently, the statistics include the min/max/mean/std of
    the actions (assumed to be in unified format) and proprio as well as the number of transitions and trajectories in the dataset.
    """
    unique_hash = hashlib.sha256(
        "".join(hash_dependencies + (str(unified_action_dim),)).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


    # fallback local path for when data_dir is not writable or not provided
    local_path = os.path.expanduser(
        os.path.join(
            "~",
            ".cache",
            "octo",
            f"dataset_statistics_{unique_hash}.json",
        )
    )

    if save_dir is not None:
        path = tf.io.gfile.join(save_dir, f"dataset_statistics_{unique_hash}.json")
    else:
        path = local_path

    # check if cache file exists and load
    if tf.io.gfile.exists(path) and not force_recompute:
        logging.info(f"Loading existing dataset statistics from {path}.")
        with tf.io.gfile.GFile(path, "r") as f:
            metadata = json.load(f)
        return metadata

    if os.path.exists(local_path) and not force_recompute:
        logging.info(f"Loading existing dataset statistics from {local_path}.")
        with open(local_path, "r") as f:
            metadata = json.load(f)
        return metadata

    dataset = dataset.traj_map(
        lambda traj: {
            "action": traj["action"],
            **(
                {"proprio": traj["observation"]["proprio"]}
                if "proprio" in traj["observation"]
                else {}
            ),
        }
    )

    cardinality = dataset.cardinality().numpy()
    if cardinality == tf.data.INFINITE_CARDINALITY:
        raise ValueError("Cannot compute dataset statistics for infinite datasets.")

    logging.info(
        "Computing dataset statistics. This may take awhile, but should only need to happen "
        "once for each dataset."
    )
    actions = []
    proprios = []
    num_transitions = 0
    num_trajectories = 0
    for traj in tqdm.tqdm(
        dataset.iterator(),
        total=cardinality if cardinality != tf.data.UNKNOWN_CARDINALITY else None,
    ):
        actions.append(traj["action"])
        if "proprio" in traj:
            proprios.append(traj["proprio"])
        num_transitions += traj["action"].shape[0]
        num_trajectories += 1
    actions = np.concatenate(actions)
    
    metadata = {
        "action": {
            "mean": actions.mean(0).tolist(),
            "std": actions.std(0).tolist(),
            "max": actions.max(0).tolist(),
            "min": actions.min(0).tolist(),
            "p99": np.quantile(actions, 0.99, 0).tolist(),
            "p01": np.quantile(actions, 0.01, 0).tolist(),
        },
        "num_transitions": num_transitions,
        "num_trajectories": num_trajectories,
    }
    if proprios:
        proprios = np.concatenate(proprios)
        metadata["proprio"] = {
            "mean": proprios.mean(0).tolist(),
            "std": proprios.std(0).tolist(),
            "max": proprios.max(0).tolist(),
            "min": proprios.min(0).tolist(),
            "p99": np.quantile(proprios, 0.99, 0).tolist(),
            "p01": np.quantile(proprios, 0.01, 0).tolist(),
        }

    try:
        with tf.io.gfile.GFile(path, "w") as f:
            json.dump(metadata, f)
    except tf.errors.PermissionDeniedError:
        logging.warning(
            f"Could not write dataset statistics to {path}. "
            f"Writing to {local_path} instead."
        )
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            json.dump(metadata, f)

    return metadata


def combine_dataset_statistics(
    all_dataset_statistics: Sequence[dict],
) -> dict:
    """Merges dataset statistics from multiple datasets."""
    merge_stat_keys = ["action", "proprio"]

    num_trajectories = [stat["num_trajectories"] for stat in all_dataset_statistics]
    num_transitions = [stat["num_transitions"] for stat in all_dataset_statistics]
    stat_weights = [
        transitions / sum(num_transitions) for transitions in num_transitions
    ]

    combined_dataset_statistics = {}
    for key in merge_stat_keys:
        combined_mean = np.array(
            [
                stat[key]["mean"] * w
                for stat, w in zip(all_dataset_statistics, stat_weights)
            ]
        ).sum(0)
        # compute combined_std for denominator `n` instead of `n-1` since numpy uses that by default for std
        # https://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
        combined_std = np.sqrt(
            np.array(
                [
                    n * np.array(stat[key]["std"]) ** 2
                    + n * (np.array(stat[key]["mean"]) - combined_mean) ** 2
                    for stat, n in zip(all_dataset_statistics, num_transitions)
                ]
            ).sum(0)
            / sum(num_transitions)
        )
        combined_dataset_statistics[key] = {
            "min": np.array([stat[key]["min"] for stat in all_dataset_statistics])
            .min(0)
            .tolist(),
            "max": np.array([stat[key]["max"] for stat in all_dataset_statistics])
            .max(0)
            .tolist(),
            "mean": combined_mean.tolist(),
            "std": combined_std.tolist(),
        }

    combined_dataset_statistics["num_trajectories"] = num_trajectories
    combined_dataset_statistics["num_transitions"] = num_transitions
    return combined_dataset_statistics


def convert_to_float32(tensor):
    if tensor.dtype != tf.float32:
        return tf.cast(tensor, tf.float32)
    return tensor

def get_non_zero_dims(tensor):
    """
    Determine the number of non-zero dimensions in the tensor.
    """
    non_zero_mask = tf.not_equal(tf.reduce_sum(tf.abs(tensor), axis=0), 0)
    return tf.reduce_sum(tf.cast(non_zero_mask, tf.int32))

def tensor_to_scalar(tensor):
    """Safely convert a tensor to a scalar value."""
    if isinstance(tensor, (int, float)):
        return tensor
    elif isinstance(tensor, tf.Tensor):
        return tf.get_static_value(tensor) or int(tensor)
    else:
        raise TypeError(f"Expected int, float, or Tensor, got {type(tensor)}")

def get_non_zero_dims(tensor):
    """
    Determine the number of non-zero dimensions in the tensor.
    Returns a scalar integer.
    """
    non_zero_mask = tf.not_equal(tf.reduce_sum(tf.abs(tensor), axis=0), 0)
    return tensor_to_scalar(tf.reduce_sum(tf.cast(non_zero_mask, tf.int32)))


def normalize_unified_action(x, metadata_key, metadata, normalization_type):
    try:
        x = tf.cast(x, tf.float32)
        
        # Log shapes and types for debugging
        logging.info(f"Input x shape: {x.shape}, type: {type(x)}")
        logging.info(f"metadata[{metadata_key}]['p01'] shape: {metadata[metadata_key]['p01'].shape}, type: {metadata[metadata_key]['p01'].dtype}")
        
        # Check if x is a SymbolicTensor
        # Check if x is a SparseTensor
        if isinstance(x, tf.SparseTensor):
            # Get non-zero indices and values from the sparse tensor
            non_zero_indices = x.indices
            non_zero_values = x.values
        else:
            # If x is a dense tensor, treat all values as non-zero
            non_zero_indices = tf.where(tf.not_equal(x, 0))
            non_zero_values = tf.gather_nd(x, non_zero_indices)
        
        # Log non-zero indices and values before normalization
        logging.info(f"Non-zero indices before normalization: {non_zero_indices}")
        logging.info(f"Non-zero values before normalization: {non_zero_values}")
        
        # Ensure metadata tensors are the right shape
        metadata_p01 = tf.cast(metadata[metadata_key]['p01'], tf.float32)
        metadata_p99 = tf.cast(metadata[metadata_key]['p99'], tf.float32)
        
        # Normalize only the non-zero values based on their indices
        if normalization_type == NormalizationType.NORMAL:
            metadata_mean = tf.cast(metadata[metadata_key]['mean'], tf.float32)
            metadata_std = tf.cast(metadata[metadata_key]['std'], tf.float32)
            norm_values = (tf.gather(non_zero_values, non_zero_indices[:, -1]) - tf.gather(metadata_mean, non_zero_indices[:, -1])) / (tf.gather(metadata_std, non_zero_indices[:, -1]) + 1e-8)
        elif normalization_type == NormalizationType.BOUNDS:
            norm_values = tf.clip_by_value(
                2 * (tf.gather(non_zero_values, non_zero_indices[:, -1]) - tf.gather(metadata_p01, non_zero_indices[:, -1])) / (tf.gather(metadata_p99, non_zero_indices[:, -1]) - tf.gather(metadata_p01, non_zero_indices[:, -1]) + 1e-8) - 1,
                -1, 1
            )
        else:
            raise ValueError(f"Unknown normalization type {normalization_type}")

        # Log normalized values
        logging.info(f"Normalized values: {norm_values}")

        # Create a new tensor with normalized values
        normalized_x = tf.tensor_scatter_nd_update(tf.zeros_like(x), non_zero_indices, norm_values)

        # Log the normalized tensor
        logging.info(f"Normalized x: {normalized_x}")

        return normalized_x
    except Exception as e:
        logging.error(f"Error in normalize_unified_action: {str(e)}")
        raise

def normalize_action_and_proprio(traj: dict, metadata: dict, normalization_type: NormalizationType):
    keys_to_normalize = {
        "action": "action",
    }
    if "proprio" in traj["observation"]:
        keys_to_normalize["proprio"] = "observation/proprio"

    try:
        for key, traj_key in keys_to_normalize.items():
            if key == "action":
                traj[traj_key] = normalize_unified_action(traj[traj_key], key, metadata, normalization_type)
            else:
                # Handling for non-action keys (e.g., proprio) remains the same
                mask = metadata[key].get("mask", tf.ones_like(metadata[key]["mean"], dtype=tf.bool))
                if normalization_type == NormalizationType.NORMAL:
                    traj[traj_key] = tf.where(
                        mask,
                        (tf.cast(traj[traj_key], tf.float32) - tf.cast(metadata[key]["mean"], tf.float32)) / 
                        (tf.cast(metadata[key]["std"], tf.float32) + 1e-8),
                        traj[traj_key]
                    )
                elif normalization_type == NormalizationType.BOUNDS:
                    traj[traj_key] = tf.where(
                        mask,
                        tf.clip_by_value(
                            2 * (tf.cast(traj[traj_key], tf.float32) - tf.cast(metadata[key]["p01"], tf.float32)) /
                            (tf.cast(metadata[key]["p99"], tf.float32) - tf.cast(metadata[key]["p01"], tf.float32) + 1e-8) - 1,
                            -1, 1
                        ),
                        traj[traj_key]
                    )
    except Exception as e:
        logging.error(f"Error in normalize_action_and_proprio: {str(e)}")
        raise

    return traj


def binarize_gripper_actions(actions: tf.Tensor, open_boundary: float = 0.95, close_boundary: float = 0.05) -> tf.Tensor:
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near
    0.0). As it transitions between the two, it sometimes passes through a few intermediate values. We relabel
    those intermediate values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel
    that chunk of intermediate values as the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > open_boundary # 0.95
    closed_mask = actions < close_boundary # 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, actions.dtype)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, actions.dtype),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )
    return new_actions


def rel_open_or_closed(actions: tf.Tensor):
    """
    Returns the initial absolute gripper state, given relative actions (-1 for closing, +1 for opening)
    Returns 1 if the gripper is initially open, 0 if it is initially closed.
    If nothing taken, assumes gripper is initially open.

    """
    opening_mask = actions > 1e-3
    closing_mask = actions < -1e-3
    old_state_mask = tf.where(opening_mask, -1, tf.where(closing_mask, -1, 0))
    # old_state_mask is 1 if closing, -1 if opening, 0 if no change

    def scan_fn(carry, i):
        return tf.cond(
            old_state_mask[i] == 0,
            lambda: tf.cast(carry, tf.float32),
            lambda: (tf.cast(old_state_mask[i], tf.float32) + 1) / 2,
        )

    return tf.scan(
        scan_fn,
        tf.range(tf.shape(actions)[0]),
        tf.zeros_like(actions[-1]),
        reverse=True,
    )[0]


def rel2abs_gripper_actions(actions: tf.Tensor):
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute gripper actions
    (0 for closed, 1 for open). Assumes that the first relative gripper is not redundant
    (i.e. close when already closed).
    """
    opening_mask = actions < -0.1
    closing_mask = actions > 0.1

    # -1 for closing, 1 for opening, 0 for no change
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(
            thresholded_actions[i] == 0,
            lambda: carry,
            lambda: thresholded_actions[i],
        )

    # if no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)
    # -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)

    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5
    return new_actions


def invert_gripper_actions(actions: tf.Tensor):
    return 1 - actions


def relabel_actions(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Relabels the actions to use the reached proprio instead. Discards the last timestep of the
    trajectory (since we don't have a next state to compute the action.)
    """
    # relabel the first 6 action dims (xyz position, xyz rotation) using the reached proprio
    movement_actions = (
        traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    )

    # discard the last timestep of the trajectory
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)

    # recombine to get full actions
    traj_truncated["action"] = tf.concat(
        [movement_actions, traj["action"][:-1, -1:]],
        axis=1,
    )

    return traj_truncated


def allocate_threads(n: Optional[int], weights: np.ndarray):
    """Allocates an integer number of threads across datasets based on weights. The final array sums to `n`,
    but each element is no less than 1. If `n` is None, then every dataset is assigned a value of AUTOTUNE.
    """
    if n is None:
        return np.array([tf.data.AUTOTUNE] * len(weights))

    assert np.all(weights >= 0), "Weights must be non-negative"
    assert (
        len(weights) <= n
    ), "Number of threads must be at least as large as length of weights"
    weights = np.array(weights) / np.sum(weights)

    allocation = np.zeros_like(weights, dtype=int)
    while True:
        # give the remaining elements that would get less than 1 a 1
        mask = (weights * n < 1) & (weights > 0)
        if not mask.any():
            break
        n -= mask.sum()
        allocation += mask.astype(int)
        # recompute the distribution over the remaining elements
        weights[mask] = 0
        weights = weights / weights.sum()
    # allocate the remaining elements
    fractional, integral = np.modf(weights * n)
    allocation += integral.astype(int)
    n -= integral.sum()
    for i in np.argsort(fractional)[::-1][: int(n)]:
        allocation[i] += 1
    return allocation