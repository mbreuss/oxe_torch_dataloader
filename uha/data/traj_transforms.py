"""
Trajectory transformation implementations for the data pipeline.
These transforms operate on complete trajectories rather than individual frames.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import logging
import numpy as np
import tensorflow as tf

from uha.data.utils.goal_relabeling import (
    uniform,
    uniform_and_future,
    bound_uniform,
    bound_uniform_and_future
)
from uha.data.utils import task_augmentation
from uha.data.utils.task_augmentation import (
    delete_and_rephrase,
    delete_task_conditioning
)
from uha.data.utils.spec import ModuleSpec

logger = logging.getLogger(__name__)

# Dictionary mapping strategy names to functions
GOAL_RELABELING_STRATEGIES = {
    'uniform': uniform,
    'uniform_and_future': uniform_and_future,
    'bound_uniform': bound_uniform,
    'bound_uniform_and_future': bound_uniform_and_future
}



class DataValidator:
    """Validates data consistency throughout the pipeline."""
    
    @staticmethod
    def validate_language_data(traj: Dict[str, Any], stage: str = "unknown") -> Dict[str, Any]:
        """Validate and ensure language data consistency."""
        tf.print(f"\n=== Language Data Validation [{stage}] ===")
        
        # Track original state
        has_task = "task" in traj
        has_language = has_task and "language_instruction" in traj["task"]
        
        tf.print("Has task:", has_task)
        tf.print("Has language:", has_language)
        
        if has_language:
            tf.print("Language type:", traj["task"]["language_instruction"].dtype)
            tf.print("Language shape:", tf.shape(traj["task"]["language_instruction"]))
        
        # Ensure task dictionary exists
        if not has_task:
            traj["task"] = {}
            tf.print("Created missing task dictionary")
            
        # Ensure language instruction exists
        if not has_language:
            tf.print("WARNING: Missing language instruction")
            traj["task"]["language_instruction"] = tf.constant("")
            
        return traj
        
    @staticmethod
    def validate_pad_masks(traj: Dict[str, Any], stage: str = "unknown") -> Dict[str, Any]:
        """Validate and ensure pad mask consistency."""
        tf.print(f"\n=== Pad Mask Validation [{stage}] ===")
        
        # Track pad mask state
        has_obs_masks = "pad_mask_dict" in traj.get("observation", {})
        has_task_masks = "pad_mask_dict" in traj.get("task", {})
        
        tf.print("Has observation masks:", has_obs_masks)
        tf.print("Has task masks:", has_task_masks)
        
        # Ensure pad mask dictionaries exist
        if not has_obs_masks:
            traj["observation"]["pad_mask_dict"] = {}
            tf.print("Created missing observation pad mask dictionary")
            
        if not has_task_masks:
            traj["task"]["pad_mask_dict"] = {}
            tf.print("Created missing task pad mask dictionary")
            
        return traj



def apply_trajectory_transforms(
    dataset,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: Dict[str, Any] = None,
    window_size: int = 1,
    action_horizon: int = 1,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: Dict[str, Any] = None,
    max_action_dim: Optional[int] = None,
    max_proprio_dim: Optional[int] = None,
    post_chunk_transforms: List[ModuleSpec] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE
) -> tf.data.Dataset:
    """Apply a sequence of trajectory transformations to the dataset."""
    tf.print("\nStarting trajectory transforms")
    validator = DataValidator()

    goal_relabeling_kwargs = goal_relabeling_kwargs or {}
    task_augment_kwargs = task_augment_kwargs or {}
    post_chunk_transforms = post_chunk_transforms or []
    
    # Add padding masks with validation
    def add_validated_pad_masks(traj: Dict[str, Any]) -> Dict[str, Any]:
        """Create pad masks with proper handling of nested structures."""
        tf.print("\n=== Creating Pad Masks ===")
        
        def create_mask_for_value(value, traj_len: int) -> tf.Tensor:
            """Create appropriate mask for a given value."""
            # Handle nested dictionaries
            if isinstance(value, dict):
                return tf.ones([traj_len], dtype=tf.bool)
                
            # Handle tensors and arrays
            if isinstance(value, (tf.Tensor, np.ndarray)):
                if value.dtype == tf.string:
                    return tf.strings.length(value) != 0
                return tf.ones([traj_len], dtype=tf.bool)
                
            # Default case
            return tf.ones([traj_len], dtype=tf.bool)

        # Debug current state
        tf.print("Trajectory keys:", list(traj.keys()))
        tf.print("Observation keys:", list(traj["observation"].keys()))
        tf.print("Task keys:", list(traj["task"].keys()))
        
        traj_len = tf.shape(traj["action"])[0]
        
        # Create observation pad masks
        obs_pad_mask_dict = {}
        for key, value in traj["observation"].items():
            try:
                obs_pad_mask_dict[key] = create_mask_for_value(value, traj_len)
                tf.print(f"Created observation mask for {key}")
            except Exception as e:
                tf.print(f"Error creating observation mask for {key}: {str(e)}")
                obs_pad_mask_dict[key] = tf.ones([traj_len], dtype=tf.bool)
        
        # Create task pad masks
        task_pad_mask_dict = {}
        for key, value in traj["task"].items():
            if key == "pad_mask_dict":
                continue
                
            try:
                task_pad_mask_dict[key] = create_mask_for_value(value, traj_len)
                tf.print(f"Created task mask for {key}")
            except Exception as e:
                tf.print(f"Error creating task mask for {key}: {str(e)}")
                task_pad_mask_dict[key] = tf.ones([traj_len], dtype=tf.bool)
        
        # Add pad masks to trajectory
        traj["observation"]["pad_mask_dict"] = obs_pad_mask_dict
        traj["task"]["pad_mask_dict"] = task_pad_mask_dict
        
        return traj
    # Add padding masks
    # Add padding masks with debugging
    # Add pad masks
    dataset = dataset.traj_map(
        add_validated_pad_masks,
        num_parallel_calls=num_parallel_calls
    )

    # Skip unlabeled data if specified
    if skip_unlabeled:
        def check_language_validity(traj):
            # Safe access of pad masks
            if "task" not in traj or "pad_mask_dict" not in traj["task"]:
                return tf.constant(True)
                
            pad_masks = traj["task"]["pad_mask_dict"]
            if "language_instruction" not in pad_masks:
                return tf.constant(True)
                
            return tf.reduce_any(pad_masks["language_instruction"])
            
        dataset = dataset.filter(check_language_validity)

    # Apply goal relabeling if specified
    if goal_relabeling_strategy and train:
        if goal_relabeling_strategy not in GOAL_RELABELING_STRATEGIES:
            raise ValueError(f"Unknown goal relabeling strategy: {goal_relabeling_strategy}")
            
        relabel_fn = GOAL_RELABELING_STRATEGIES[goal_relabeling_strategy]
        dataset = dataset.traj_map(
            lambda traj: relabel_fn(traj, **goal_relabeling_kwargs),
            num_parallel_calls=num_parallel_calls
        )
    
    dataset = dataset.traj_map(
        add_validated_pad_masks,
        num_parallel_calls=num_parallel_calls
    )

    # Skip unlabeled data if specified
    if skip_unlabeled:
        def check_language_valid(traj):
            # Validate before checking
            traj = validator.validate_language_data(traj, "filter-check")
            traj = validator.validate_pad_masks(traj, "filter-check")
            
            # Safe access with defaults
            has_mask = tf.py_function(
                lambda: "language_instruction" in traj["task"]["pad_mask_dict"],
                [], 
                tf.bool
            )
            
            def check_mask():
                return tf.reduce_any(traj["task"]["pad_mask_dict"]["language_instruction"])
            
            def default_value():
                return tf.constant(True)
            
            return tf.cond(has_mask, check_mask, default_value)
            
        dataset = dataset.filter(check_language_valid)

    # Validate after each transform stage
    def validate_transform_result(traj: Dict[str, Any], stage: str) -> Dict[str, Any]:
        traj = validator.validate_language_data(traj, stage)
        traj = validator.validate_pad_masks(traj, stage)
        return traj

    # Apply remaining transforms with validation
    if goal_relabeling_strategy and train:
        relabel_fn = getattr(goal_relabeling, goal_relabeling_strategy)
        dataset = dataset.traj_map(
            lambda traj: validate_transform_result(
                relabel_fn(traj, **goal_relabeling_kwargs),
                "post-relabel"
            )
        )

    # Apply subsampling if specified
    if subsample_length is not None:
        dataset = dataset.traj_map(
            lambda traj: subsample(traj, subsample_length),
            num_parallel_calls=num_parallel_calls
        )

    # Clip values if specified
    if max_action is not None or max_proprio is not None:
        def clip_fn(traj):
            if max_action is not None:
                traj["action"] = tf.clip_by_value(traj["action"], -max_action, max_action)
            if max_proprio is not None and "proprio" in traj["observation"]:
                traj["observation"]["proprio"] = tf.clip_by_value(
                    traj["observation"]["proprio"], -max_proprio, max_proprio
                )
            return traj
        dataset = dataset.traj_map(clip_fn, num_parallel_calls=num_parallel_calls)

    # Pad dimensions if specified
    if max_action_dim is not None or max_proprio_dim is not None:
        dataset = dataset.traj_map(
            lambda traj: pad_actions_and_proprio(
                traj,
                max_action_dim=max_action_dim,
                max_proprio_dim=max_proprio_dim
            ),
            num_parallel_calls=num_parallel_calls
        )

    # Skip unlabeled data if specified and language instructions exist
    if skip_unlabeled:
        def has_valid_language(traj):
            # Use tensorflow conditionals to handle missing keys
            has_task = tf.py_function(lambda: "task" in traj, [], tf.bool)
            has_pad_mask = tf.py_function(
                lambda: "pad_mask_dict" in traj.get("task", {}), 
                [], 
                tf.bool
            )
            has_lang_mask = tf.py_function(
                lambda: "language_instruction" in traj.get("task", {}).get("pad_mask_dict", {}),
                [],
                tf.bool
            )
            
            def check_mask():
                return tf.reduce_any(traj["task"]["pad_mask_dict"]["language_instruction"])
                
            def default_value():
                return tf.constant(True)
                
            return tf.cond(
                tf.logical_and(has_task, tf.logical_and(has_pad_mask, has_lang_mask)),
                check_mask,
                default_value
            )
            
        dataset = dataset.filter(has_valid_language)

    # Apply chunking
    dataset = dataset.traj_map(
        lambda traj: chunk_act_obs(
            traj,
            window_size=window_size,
            action_horizon=action_horizon
        ),
        num_parallel_calls=num_parallel_calls
    )

    # Apply any additional post-chunk transforms
    for transform_spec in post_chunk_transforms:
        transform_fn = ModuleSpec.instantiate(transform_spec)
        dataset = dataset.traj_map(
            transform_fn,
            num_parallel_calls=num_parallel_calls
        )

    return dataset


@dataclass
class ChunkConfig:
    """Configuration for trajectory chunking."""
    window_size: int = 1
    action_horizon: int = 1


@dataclass
class PaddingConfig:
    """Configuration for padding operations."""
    max_action_dim: Optional[int] = None
    max_proprio_dim: Optional[int] = None


class PadMaskGenerator:
    """Handles creation and management of padding masks."""

    @staticmethod
    def create_pad_mask_dict(traj: Dict[str, Any]) -> Dict[str, Dict[str, tf.Tensor]]:
        """Create padding mask dictionary for trajectory."""
        traj_len = tf.shape(traj["action"])[0]
        pad_masks = {}
        
        for key in ["observation", "task"]:
            pad_mask_dict = {}
            for subkey in traj[key]:
                if traj[key][subkey].dtype == tf.string:
                    # For language instructions, images, and depth
                    pad_mask_dict[subkey] = tf.strings.length(traj[key][subkey]) != 0
                else:
                    # Other keys aren't padding
                    pad_mask_dict[subkey] = tf.ones([traj_len], dtype=tf.bool)
            pad_masks[key] = pad_mask_dict
            
        return pad_masks


class DimensionPadder:
    """Handles padding operations for actions and proprio."""
    
    @staticmethod
    def pad_actions_and_proprio(
            traj: Dict[str, Any],
            config: PaddingConfig
    ) -> Dict[str, Any]:
        """Pad actions and proprio to specified dimensions."""
        # Initialize action mask
        traj["action_pad_mask"] = tf.ones_like(traj["action"], dtype=tf.bool)
        
        # Pad actions if needed
        if config.max_action_dim is not None:
            action_dim = traj["action"].shape[-1]
            if action_dim > config.max_action_dim:
                raise ValueError(
                    f"action_dim ({action_dim}) > max_action_dim ({config.max_action_dim})"
                )
            
            padding = [[0, 0]] * (len(traj["action"].shape) - 1) + \
                     [[0, config.max_action_dim - action_dim]]
                     
            for key in ["action", "action_pad_mask"]:
                traj[key] = tf.pad(traj[key], padding)

        # Pad proprio if needed
        if config.max_proprio_dim is not None and "proprio" in traj["observation"]:
            proprio_dim = traj["observation"]["proprio"].shape[-1]
            if proprio_dim > config.max_proprio_dim:
                raise ValueError(
                    f"proprio_dim ({proprio_dim}) > max_proprio_dim ({config.max_proprio_dim})"
                )
                
            padding = [[0, 0], [0, config.max_proprio_dim - proprio_dim]]
            traj["observation"]["proprio"] = tf.pad(
                traj["observation"]["proprio"],
                padding
            )

        return traj


class TrajectoryChunker:
    """Handles trajectory chunking operations."""
    
    def __init__(self, config: ChunkConfig):
        self.config = config

    def chunk_trajectory(self, traj: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk trajectory into windows."""
        traj_len = tf.shape(traj["action"])[0]

        # Create history indices
        history_indices = tf.range(traj_len)[:, None] + tf.range(
            -self.config.window_size + 1, 1
        )
        
        # Create timestep padding mask
        timestep_pad_mask = history_indices >= 0
        
        # Handle boundary conditions
        history_indices = tf.maximum(history_indices, 0)
        
        # Chunk observations
        traj["observation"] = tf.nest.map_structure(
            lambda x: tf.gather(x, history_indices), 
            traj["observation"]
        )
        traj["observation"]["timestep_pad_mask"] = timestep_pad_mask

        # Chunk actions
        if len(traj["action"].shape) == 2:
            action_chunk_indices = tf.range(traj_len)[:, None] + tf.range(
                self.config.action_horizon
            )
            action_chunk_indices = tf.minimum(action_chunk_indices, traj_len - 1)
            traj["action"] = tf.gather(traj["action"], action_chunk_indices)
        else:
            if traj["action"].shape[1] < self.config.action_horizon:
                raise ValueError(
                    f"action_horizon ({self.config.action_horizon}) > "
                    f"pre-chunked action dimension ({traj['action'].shape[1]})"
                )
            traj["action"] = traj["action"][:, :self.config.action_horizon]

        # Add history axis to actions
        traj["action"] = tf.gather(traj["action"], history_indices)

        # Handle goal timesteps
        if "timestep" in traj["task"]:
            goal_timestep = traj["task"]["timestep"]
        else:
            goal_timestep = tf.fill([traj_len], traj_len - 1)

        # Compute relative goal timesteps
        t, w, h = tf.meshgrid(
            tf.range(traj_len),
            tf.range(self.config.window_size),
            tf.range(self.config.action_horizon),
            indexing="ij"
        )
        relative_goal_timestep = goal_timestep[:, None, None] - (
            t - (self.config.window_size + 1) + w + h
        )
        
        # Add task completion information
        traj["observation"]["task_completed"] = relative_goal_timestep <= 0

        # Update action padding mask
        traj["action_pad_mask"] = tf.logical_and(
            traj["action_pad_mask"][:, None, None, :]
            if len(traj["action_pad_mask"].shape) == 2
            else traj["action_pad_mask"][:, None, :],
            tf.logical_not(traj["observation"]["task_completed"])[:, :, :, None]
        )

        return traj


class SubsampleGenerator:
    """Handles trajectory subsampling operations."""
    
    @staticmethod
    def subsample_trajectory(traj: Dict[str, Any], subsample_length: int) -> Dict[str, Any]:
        """Subsample trajectory to specified length."""
        traj_len = tf.shape(traj["action"])[0]
        
        if traj_len > subsample_length:
            indices = tf.random.shuffle(tf.range(traj_len))[:subsample_length]
            return tf.nest.map_structure(lambda x: tf.gather(x, indices), traj)
            
        return traj


# Create instances of transform handlers
pad_mask_generator = PadMaskGenerator()
dimension_padder = DimensionPadder()


def chunk_act_obs(
        traj: Dict[str, Any],
        window_size: int = 1,
        action_horizon: int = 1
) -> Dict[str, Any]:
    """Chunk actions and observations in trajectory."""
    config = ChunkConfig(window_size=window_size, action_horizon=action_horizon)
    chunker = TrajectoryChunker(config)
    return chunker.chunk_trajectory(traj)


def subsample(traj: Dict[str, Any], subsample_length: int) -> Dict[str, Any]:
    """Subsample trajectory to specified length."""
    return SubsampleGenerator.subsample_trajectory(traj, subsample_length)


def add_pad_mask_dict(traj: Dict[str, Any]) -> Dict[str, Any]:
    """Create padding mask dictionary with detailed debugging."""
    traj_len = tf.shape(traj["action"])[0]
    
    tf.print("\nDEBUG: add_pad_mask_dict")
    tf.print("Trajectory keys:", list(traj.keys()))
    if "task" in traj:
        tf.print("Task keys:", list(traj["task"].keys()))
        if "language_instruction" in traj["task"]:
            tf.print("Language instruction type:", traj["task"]["language_instruction"].dtype)
    
    # Create observation pad masks
    obs_pad_mask_dict = {}
    for key in traj["observation"]:
        if isinstance(traj["observation"][key], (tf.Tensor, np.ndarray)):
            if traj["observation"][key].dtype == tf.string:
                obs_pad_mask_dict[key] = tf.strings.length(traj["observation"][key]) != 0
            else:
                obs_pad_mask_dict[key] = tf.ones([traj_len], dtype=tf.bool)
    
    # Create task pad masks with more careful handling
    task_pad_mask_dict = {}
    for key in traj["task"]:
        if key == "pad_mask_dict":  # Skip if already exists
            continue
            
        if isinstance(traj["task"][key], (tf.Tensor, np.ndarray)):
            if traj["task"][key].dtype == tf.string:
                task_pad_mask_dict[key] = tf.strings.length(traj["task"][key]) != 0
                tf.print(f"Created mask for {key}")
            else:
                task_pad_mask_dict[key] = tf.ones([traj_len], dtype=tf.bool)
                tf.print(f"Created default mask for {key}")
    
    tf.print("Created pad masks. Task pad mask keys:", list(task_pad_mask_dict.keys()))
    
    # Add pad masks to trajectory
    traj["observation"]["pad_mask_dict"] = obs_pad_mask_dict
    traj["task"]["pad_mask_dict"] = task_pad_mask_dict
    
    return traj

def has_valid_language(traj: Dict[str, Any]) -> tf.Tensor:
    """Check for valid language with safe access."""
    tf.print("\nDEBUG: has_valid_language check")
    tf.print("Task keys:", list(traj["task"].keys()))
    
    # Safe dictionary access
    if "pad_mask_dict" not in traj["task"]:
        tf.print("No pad_mask_dict in task")
        return tf.constant(True)
    
    pad_mask_dict = traj["task"]["pad_mask_dict"]
    tf.print("Pad mask dict keys:", list(pad_mask_dict.keys()))
    
    if "language_instruction" not in pad_mask_dict:
        tf.print("No language_instruction in pad_mask_dict")
        return tf.constant(True)
        
    # Check mask validity
    return tf.reduce_any(pad_mask_dict["language_instruction"])

def pad_actions_and_proprio(
        traj: Dict[str, Any],
        max_action_dim: Optional[int] = None,
        max_proprio_dim: Optional[int] = None
) -> Dict[str, Any]:
    """Pad actions and proprio to specified dimensions."""
    config = PaddingConfig(
        max_action_dim=max_action_dim,
        max_proprio_dim=max_proprio_dim
    )
    return dimension_padder.pad_actions_and_proprio(traj, config)