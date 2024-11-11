from functools import partial
import json
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from uha.data import obs_transforms, traj_transforms
from uha.data.utils import goal_relabeling, task_augmentation
from uha.data.utils.data_utils import (
    allocate_threads,
    get_dataset_statistics,
    NormalizationType,
    normalize_action_and_proprio,
    pprint_data_mixture,
    sample_match_keys_uniform,
    
    tree_map,
)
from uha.data.utils.spec import ModuleSpec
from uha.data.utils.rlds_utils import *

from uha.data.oxe.oxe_standardization_transforms import (
    get_action_space_index,
)

def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    window_size: int = 1,
    action_horizon: int = 1,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    max_action_dim: Optional[int] = None,
    max_proprio_dim: Optional[int] = None,
    post_chunk_transforms: Sequence[ModuleSpec] = (),
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of
    "relabeling" (e.g. filtering, chunking, adding goals, dropping keys). Transforms that happen in this
    function should have the following properties:

    - They require access to an entire trajectory (i.e. they cannot be applied in a frame-wise manner).
    - They are generally not CPU-intensive, mostly involving moving and copying data.
    - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        window_size (int, optional): The window size to chunk both observations and actions into.
        action_horizon (int, optional): The size of the action chunk (present and future actions) to include in
            the chunked actions.
        subsample_length (int, optional): If provided, trajectories longer than this will be subsampled to
            this length (after goal relabeling and chunking).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        task_augment_strategy (str, optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augment_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
            function.
        max_action_dim (int, optional): If provided, datasets with an action dimension less than this will be
            padded to this dimension.
        max_proprio_dim (int, optional): If provided, datasets with a proprio dimension less than this will be
            padded to this dimension.
        post_chunk_transforms (Sequence[ModuleSpec]): ModuleSpecs of trajectory transforms applied after
            chunking.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError(
                "skip_unlabeled=True but dataset does not have language labels."
            )
        dataset = dataset.filter(
            lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != "")
        )

    if max_action is not None:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action)
        )

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(
            lambda x: tf.math.reduce_all(
                tf.math.abs(x["observation"]["proprio"]) <= max_proprio
            )
        )

    # marks which entires of the observation and task dicts are padding
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    # optionally pads actions and proprio to a consistent number of dimensions
    dataset = dataset.traj_map(
        partial(
            traj_transforms.pad_actions_and_proprio,
            max_action_dim=max_action_dim,
            max_proprio_dim=max_proprio_dim,
        ),
        num_parallel_calls,
    )

    # updates the "task" dict
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(
                getattr(goal_relabeling, goal_relabeling_strategy),
                **goal_relabeling_kwargs,
            ),
            num_parallel_calls,
        )

    # must run task augmentation before chunking, in case it changes goal timesteps
    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks observations and actions
    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            action_horizon=action_horizon,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )

    for transform_spec in post_chunk_transforms:
        dataset = dataset.traj_map(
            ModuleSpec.instantiate(transform_spec),
            num_parallel_calls,
        )

    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[dict, Mapping[str, dict]] = {},
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    resize_size_future_obs: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    image_dropout_prob: float = 0.0,
    image_dropout_keep_key: Optional[str] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """Applies common transforms that happen at a frame level. These transforms are usually more
    CPU-intensive, (e.g. decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a dict of
            dicts is provided, then key "k" will be used for "image_{k}" (names determined by `image_obs_keys`
            in `make_dataset_from_rlds`). Augmentation will be skipped for missing keys (so pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        resize_size_future_obs (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): Same as resize_size, but for depth
            images.
        image_dropout_prob (float): Probability of dropping out images, applied to each image key
            independently. At least one image will always be present.
        image_dropout_keep_key (str, optional): Optionally provide a key to always keep during image dropout
            for example for image observations that are essential for action prediction.
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    # convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[dict], dict], frame: dict) -> dict:
        # task is not chunked -- apply fn directly
        frame["task"] = fn(frame["task"])
        # observation is chunked -- apply fn along first axis
        frame["observation"] = dl.vmap(fn)(frame["observation"])

        return frame
    
    def apply_obs_transform_future(fn: Callable[[dict], dict], frame: dict) -> dict:
        if "future_obs" in frame:
            frame["future_obs"] = dl.vmap(fn)(frame["future_obs"])
        return frame

    # decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(
                obs_transforms.decode_and_resize,
                resize_size=resize_size,
                depth_resize_size=depth_resize_size,
            ),
        ),
        num_parallel_calls,
    )
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform_future,
            partial(
                obs_transforms.decode_and_resize,
                resize_size=resize_size_future_obs,
                depth_resize_size=depth_resize_size,
            ),
        ),
        num_parallel_calls,
    )

    if train:
        # augment all images with the same seed, skipping padding images
        def aug_and_dropout(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            dropout_fn = partial(
                obs_transforms.image_dropout,
                seed=seed,
                dropout_prob=image_dropout_prob,
                always_keep_key=image_dropout_keep_key,
            )
            aug_fn = partial(
                obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs
            )
            frame = apply_obs_transform(dropout_fn, frame)
            frame = apply_obs_transform(aug_fn, frame)
            return frame

        dataset = dataset.frame_map(aug_and_dropout, num_parallel_calls)

    return dataset


def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[ModuleSpec] = None,
    shuffle: bool = True,
    image_obs_keys: Mapping[str, Optional[str]] = {},
    depth_obs_keys: Mapping[str, Optional[str]] = {},
    proprio_obs_key: Optional[str] = None,
    language_key: Optional[str] = None,
    language_key_NILS: Optional[str] = None,
    Gt_ann_dirs: Optional[list] = None,
    NILS_ann_dirs: Optional[list] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    force_recompute_dataset_statistics: bool = False,
    action_normalization_mask: Optional[Sequence[bool]] = None,
    filter_functions: Sequence[ModuleSpec] = (),
    skip_norm: bool = False,
    ignore_errors: bool = False,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    dataset_size_limit: int = None,
) -> Tuple[dl.DLataset, dict]:
    """This function is responsible for loading a specific RLDS dataset from storage and getting it into a
    standardized format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the
    trajectory into a standard format, which includes the keys "observation" and "action". "observation"
    should be a dictionary containing some number of additional keys, which will be extracted into an even
    more standardized format according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in
    place of an old name to insert padding. For example, if after `standardize_fn`, your "observation" dict
    has RGB images called "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary":
    None, "wrist": "wrist"}`, then the resulting dataset will have an "observation" dict containing the keys
    "image_primary", "image_secondary", and "image_wrist", where "image_primary" corresponds to "workspace",
    "image_secondary" is a padding image, and "image_wrist" corresponds to "wrist".

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will
    contain the key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset,
            since one file usually contains many trajectories!).
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to
            extract from the "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in
            image_obs_keys.items()}`. If a value of `old` is None, inserts a padding image instead (empty
            string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        proprio_obs_key (str, optional): If provided, the "obs" dict will contain the key "proprio", extracted from
            `traj["observation"][proprio_obs_key]`.
        language_key (str, optional): If provided, the "task" dict will contain the key
            "language_instruction", extracted from `traj[language_key]`. If language_key fnmatches multiple
            keys, we sample one uniformly.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. May also provide "num_transitions" and "num_trajectories" keys for downstream usage
            (e.g., for `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        force_recompute_dataset_statistics (bool, optional): If True and `dataset_statistics` is None, will
            recompute the dataset statistics regardless of whether they are already cached.
        action_normalization_mask (Sequence[bool], optional): If provided, only normalizes action dimensions
            where the corresponding mask is True. For example, you might not want to normalize the gripper
            action dimension if it's always exactly 0 or 1. By default, all action dimensions are normalized.
        filter_functions (Sequence[ModuleSpec]): ModuleSpecs for filtering functions applied to the
            raw dataset.
        skip_norm (bool): If true, skips normalization of actions and proprio. Default: False.
        ignore_errors (bool): If true, skips erroneous dataset elements via dataset.ignore_errors(). Default: False.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action"}

    def restructure(traj: dict) -> dict:
        """Restructure trajectory into standardized format."""
        if standardize_fn is not None:
            traj = ModuleSpec.instantiate(standardize_fn)(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(f"Missing keys: {REQUIRED_KEYS - set(traj.keys())}. Check standardize_fn.")

        # Process main components
        new_obs, traj_len = RLDSProcessing.process_observations(
            traj, image_obs_keys, depth_obs_keys, proprio_obs_key
        )
        task, local_lang_key = RLDSProcessing.process_language(
            traj, traj_len, Gt_ann_dirs, NILS_ann_dirs, language_key, language_key_NILS
        )
        RLDSProcessing.add_language_metadata(task, local_lang_key, traj_len, language_key)
        
        # Process robot information
        robot_information = RLDSProcessing.process_robot_information(traj["observation"], traj_len)
        task['robot_information'] = robot_information

        # Process action space index
        try:
            task['action_space_index'] = tf.cast(
                convert_scalar_to_1d(traj['action_space_index'], task['robot_information']), 
                tf.int32
            )
        except Exception as e:
            logger.warning(f"Failed to process action_space_index: {e}")
        try:
            frequency = RLDSProcessing.process_frequency(traj, traj_len)
            task['frequency'] = frequency
            print(f'Frequency: {frequency}')
            # Validate shape matches
            tf.debugging.assert_equal(
                tf.shape(frequency)[0],
                traj_len,
                message="frequency shape should match trajectory length"
            )
        except Exception as e:
            logger.warning(f"Failed to process frequency: {e}")
            # Optionally provide a default value if processing fails
            task['frequency'] = tf.repeat(1, traj_len)  # Default to 1Hz if processing fails
        return {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
        }

    # Main execution flow starts here
    name = name + ":0.1.0" if name == "bc_z" else 'droid' if name == 'eef_droid' else name
    builder = tfds.builder(name, data_dir=data_dir)
    
    # Handle dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        dataset_statistics = compute_dataset_statistics(
            builder, filter_functions, ignore_errors, restructure,
            proprio_obs_key, standardize_fn, force_recompute_dataset_statistics
        )
    
    dataset_statistics = tree_map(np.array, dataset_statistics)
    DatasetUtils.validate_normalization_mask(action_normalization_mask, dataset_statistics)

    # Build and process dataset
    split = DatasetUtils.get_dataset_split(builder, train, dataset_size_limit)
    dataset = dl.DLataset.from_rlds(
        builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads
    )
    
    # Apply filters and transformations
    for filter_fcn_spec in filter_functions:
        dataset = dataset.filter(ModuleSpec.instantiate(filter_fcn_spec))
    
    if ignore_errors:
        dataset = dataset.ignore_errors()

    dataset = dataset.traj_map(restructure, num_parallel_calls).filter(
        lambda traj: tf.shape(traj["action"])[0] > 0
    )

    if not skip_norm:
        dataset = dataset.traj_map(
            partial(normalize_action_and_proprio,
                   metadata=dataset_statistics,
                   normalization_type=action_proprio_normalization_type),
            num_parallel_calls,
        )
    else:
        logger.warning("Dataset normalization disabled (skip_norm=True)")

    return dataset, dataset_statistics


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    batch_size: Optional[int] = None,
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        train: whether this is a training or validation dataset.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    dataset_len = dataset_statistics["num_transitions"]

    # sequential batch (parallel batch seems to use much more memory)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    dataset.dataset_statistics = dataset_statistics
    dataset.dataset_len = dataset_len
    return dataset


def make_interleaved_dataset(
    dataset_kwargs_list: Sequence[dict],
    sample_weights: Optional[Sequence[float]] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    """Creates an interleaved dataset from list of dataset kwargs. Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overidden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overidden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # default to uniform sampling
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)
    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(
            f"sample_weights must be None or have length {len(dataset_kwargs_list)}."
        )

    # go through datasets once to get sizes
    dataset_sizes = []
    all_dataset_statistics = {}
    for dataset_kwargs in dataset_kwargs_list:
        _, dataset_statistics = make_dataset_from_rlds(**dataset_kwargs, train=train)
        dataset_sizes.append(dataset_statistics["num_transitions"])
        assert (
            dataset_kwargs["name"] not in all_dataset_statistics
        ), f"Duplicate name {dataset_kwargs['name']}"
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    # Get the indices of the "primary" datasets (i.e., datasets with sample_weight >= 1.0)
    primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) 
                                    if sample_weights[idx] >= 1.0], dtype=np.int64)

    # balance and normalize weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # Effective Dataset Length calculation
    if len(primary_dataset_indices) > 0:
        dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())
    else:
        # Fallback when no primary datasets exist
        dataset_len = int((np.array(dataset_sizes) / sample_weights).max())
    # Or alternatively, you could use a default value:
    # dataset_len = max(dataset_sizes[0])  # or some other default
    # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
    #   =>> Note :: Only counting the "primary" datasets (i.e., datasets with sample_weight >= 1.0)
    # dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())
    # dataset_len = int((np.array(dataset_sizes) / sample_weights)[primary_dataset_indices].max())
    # allocate threads based on weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    logging.info("Threads per dataset: %s", threads_per_dataset)
    logging.info("Reads per dataset: %s", reads_per_dataset)

    # construct datasets
    datasets = []
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset, _ = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
        )
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        ).flatten(num_parallel_calls=threads)
        datasets.append(dataset)

    # interleave at the frame level and then shuffle
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(
        datasets, sample_weights
    ).shuffle(shuffle_buffer_size)

    # apply frame transforms
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # sequential batch (parallel batch seems to use much more memory)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    dataset = dataset.ignore_errors(log_warning=True)

    # save for later
    dataset.dataset_statistics = all_dataset_statistics
    dataset.sample_weights = sample_weights
    dataset.dataset_len = dataset_len
    return dataset