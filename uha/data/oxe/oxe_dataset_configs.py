"""Configuration definitions for Open X-Embodiment datasets."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Set, Union, Any, Sequence

from uha.data.utils.data_utils import filter_by_language_key
from uha.data.utils.spec import ModuleSpec


class ProprioEncoding(str, Enum):
    """Available proprioception encoding schemes."""
    NONE = "none"  # no proprio provided
    POS_EULER = "pos_euler"  # EEF XYZ + roll-pitch-yaw + gripper
    POS_QUAT = "pos_quat"  # EEF XYZ + quaternion + gripper
    JOINT = "joint"  # joint angles + gripper
    JOINT_BIMANUAL = "joint_bimanual"  # 2 x [joint angles + gripper]
    POS_NAV = "pos_nav"  # XY + yaw


class ActionEncoding(str, Enum):
    """Available action encoding schemes."""
    EEF_POS = "eef_pos"  # EEF delta XYZ + rpy + gripper
    JOINT_POS = "joint_pos"  # Joint delta position + gripper
    JOINT_POS_BIMANUAL = "joint_pos_bimanual"  # 2 x [joint pos + gripper]
    NAV_2D = "nav_2d"  # [delta_x, delta_y] waypoint
    JOINT_POS_BIMANUAL_NAV = "joint_pos_bimanual_nav"  # Bimanual + base velocity


@dataclass
class ImageConfig:
    """Configuration for image observations."""
    primary: Optional[str] = None
    secondary: Optional[str] = None
    wrist: Optional[str] = None


@dataclass
class BaseConfig:
    """Base configuration for datasets."""
    # Required fields first
    robot_name: str
    action_space: str
    proprio_encoding: ProprioEncoding
    action_encoding: ActionEncoding
    
    # Optional fields with defaults
    transform_type: str = "franka"
    num_arms: int = 1
    language_key: Optional[str] = None
    dataset_size_limit: Optional[int] = None


@dataclass
class DatasetConfig(BaseConfig):
    """Core configuration for datasets."""
    # Image configurations
    image_obs_keys: ImageConfig = field(default_factory=ImageConfig)
    depth_obs_keys: ImageConfig = field(default_factory=ImageConfig)
    
    # Optional parameters with defaults
    data_dir: Optional[Path] = None
    filter_functions: tuple = field(default_factory=tuple)
    standardize_fn: Optional[ModuleSpec] = None
    proprio_obs_key: Optional[str] = None
    action_normalization_mask: Optional[Sequence[bool]] = None
    dataset_statistics: Optional[Dict] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.data_dir and isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir).expanduser()
        if not self.robot_name:
            raise ValueError("Robot name must be specified")
        if not self.action_space:
            raise ValueError("Action space must be specified")

    @classmethod
    def create(cls, robot_name: str, action_space: str, proprio_encoding: ProprioEncoding,
              action_encoding: ActionEncoding, **kwargs) -> 'DatasetConfig':
        """Factory method for creating dataset configs."""
        return cls(
            robot_name=robot_name,
            action_space=action_space,
            proprio_encoding=proprio_encoding,
            action_encoding=action_encoding,
            **kwargs
        )


@dataclass
class OXEDatasetConfig(BaseConfig):  # Renamed from DatasetConfig
    """Core configuration for Open X-Embodiment datasets."""
    # Image configurations
    image_obs_keys: ImageConfig = field(default_factory=ImageConfig)
    depth_obs_keys: ImageConfig = field(default_factory=ImageConfig)
    
    # Optional parameters with defaults
    data_dir: Optional[Path] = None
    filter_functions: tuple = field(default_factory=tuple)
    standardize_fn: Optional[ModuleSpec] = None
    proprio_obs_key: Optional[str] = None
    action_normalization_mask: Optional[Sequence[bool]] = None
    dataset_statistics: Optional[Dict] = None


# Define the configurations
OXE_DATASET_CONFIGS: Dict[str, OXEDatasetConfig] = {
    "bridge_dataset": DatasetConfig(
        robot_name="WindowX",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image_0", secondary="image_1", wrist=None),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        language_key="groundtruth*"
    ),

    "droid": OXEDatasetConfig(
        robot_name="Franka",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(
            primary="exterior_image_1_left",
            secondary="exterior_image_2_left",
            wrist="wrist_image_left"
        ),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.JOINT_POS,
        language_key="language_instruction*",
        dataset_size_limit=1000
    ),

    "berkeley_mvp_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="xArm",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(wrist="hand_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_QUAT,
        action_encoding=ActionEncoding.JOINT_POS,
        data_dir=Path("~/tensorflow_datasets")
    ),

    "berkeley_cable_routing": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="hand_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "libero_spatial_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),

    "fractal20220817_data": OXEDatasetConfig(
        robot_name="RT1",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_QUAT,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "MO": OXEDatasetConfig(  # Mixed Objects dataset
        robot_name="Mixed",
        action_space="mixed",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),
   
    # Kitchen datasets
    "kit_irl_real_kitchen_lang": OXEDatasetConfig(
        robot_name="Franka",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="image_top", secondary="image_side", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.JOINT_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets"),
        language_key="language_instruction*"
    ),

    "kit_irl_real_kitchen_vis": OXEDatasetConfig(
        robot_name="Franka",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="image_top", secondary="image_side", wrist=None),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.JOINT_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets")
    ),

    # ALOHA datasets
    "aloha_mobile": DatasetConfig(
        robot_name="ViperX",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="cam_high", wrist="cam_right_wrist"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT_BIMANUAL,
        action_encoding=ActionEncoding.JOINT_POS_BIMANUAL_NAV,
        num_arms=2,
        data_dir=Path("~/tensorflow_datasets")
    ),

    "aloha_static_dataset": OXEDatasetConfig(
        robot_name="ViperX",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(
            primary="cam_high",
            secondary="cam_low",
            wrist="cam_right_wrist"
        ),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT_BIMANUAL,
        action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
        num_arms=2
    ),

    "aloha_dagger_dataset": OXEDatasetConfig(
        robot_name="ViperX",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(
            primary="cam_high",
            secondary="cam_low",
            wrist="cam_right_wrist"
        ),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT_BIMANUAL,
        action_encoding=ActionEncoding.JOINT_POS_BIMANUAL,
        num_arms=2
    ),

    # RT1 and related
    "fractal20220817_data": OXEDatasetConfig(
        robot_name="RT1",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_QUAT,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # RoboTurk
    "roboturk": OXEDatasetConfig(
        robot_name="Sawyer",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="front_rgb"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.NONE,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # TACO Play
    "taco_play": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="rgb_static", wrist="rgb_gripper"),
        depth_obs_keys=ImageConfig(primary="depth_static", wrist="depth_gripper"),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # Jaco Play
    "jaco_play": OXEDatasetConfig(
        robot_name="Jaco 2",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="image_wrist"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # Navigation datasets
    "gnm_dataset": OXEDatasetConfig(
        robot_name="Navigation Robot",
        action_space="2D waypoint",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_NAV,
        action_encoding=ActionEncoding.NAV_2D
    ),

    # RoboSet
    "robo_set": OXEDatasetConfig(
        robot_name="Franka",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(
            primary="image_top",
            secondary="image_left",
            wrist="image_wrist"
        ),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.JOINT_POS
    ),

    # UCSD datasets
    "ucsd_kitchen_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="xArm",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # UT datasets
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="xArm",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(
            primary="image",
            secondary="image2",
            wrist="hand_image"
        ),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # DLR datasets
    "dlr_sara_pour_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="DLR SARA",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # CMU datasets
    "cmu_playing_with_food": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(
            primary="image",
            wrist="finger_vision_1"
        ),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # Language datasets
    "language_table": OXEDatasetConfig(
        robot_name="xArm",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="rgb"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # NYU datasets
    "nyu_door_opening_surprising_effectiveness": OXEDatasetConfig(
        robot_name="Hello Stretch",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(wrist="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.NONE,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "nyu_franka_play_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(
            primary="image",
            secondary="image_additional_view"
        ),
        depth_obs_keys=ImageConfig(
            primary="depth",
            secondary="depth_additional_view"
        ),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # Stanford datasets
    "stanford_hydra_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # LIBERO datasets
    "libero_spatial_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),

    "libero_object_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),
    "libero_goal_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),
    "libero_10_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),
}

class DatasetRegistry:
    """Registry for dataset configurations."""
    
    def __init__(self):
        self._configs: Dict[str, OXEDatasetConfig] = {}

    def get_config(self, name: str) -> OXEDatasetConfig:
        """Get configuration for a dataset."""
        if name not in self._configs:
            raise ValueError(f"No configuration found for dataset: {name}")
        return self._configs[name]

    def register_config(self, name: str, config: OXEDatasetConfig):
        """Register a new dataset configuration."""
        self._configs[name] = config

DATASET_REGISTRY = DatasetRegistry()
DATASET_REGISTRY._configs = {
    # Bridge and variants
    "bridge_dataset": OXEDatasetConfig(
        robot_name="WindowX",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image_0", secondary="image_1"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        language_key="groundtruth*"
    ),

    # Kitchen datasets
    "kit_irl_real_kitchen_lang": OXEDatasetConfig(
        robot_name="Franka",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="image_top", secondary="image_side", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.JOINT_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets"),
        language_key="language_instruction*"
    ),

    "kit_irl_real_kitchen_vis": OXEDatasetConfig(
        robot_name="Franka",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="image_top", secondary="image_side", wrist=None),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.JOINT_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets")
    ),

    # Droid dataset
    "droid": OXEDatasetConfig(
        robot_name="Franka",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="exterior_image_1_left", secondary="exterior_image_2_left", wrist="wrist_image_left"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.JOINT_POS,
        language_key="language_instruction*",
        dataset_size_limit=1000
    ),

    # Berkeley datasets
    "berkeley_mvp_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="xArm",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(wrist="hand_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_QUAT,
        action_encoding=ActionEncoding.JOINT_POS,
        data_dir=Path("~/tensorflow_datasets")
    ),

    "berkeley_cable_routing": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="hand_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "berkeley_autolab_ur5": OXEDatasetConfig(
        robot_name="UR5",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="hand_image"),
        depth_obs_keys=ImageConfig(primary="depth"),
        proprio_encoding=ProprioEncoding.POS_QUAT,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # ALOHA datasets
    "aloha_mobile": OXEDatasetConfig(
        robot_name="ViperX",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="cam_high", wrist="cam_right_wrist"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT_BIMANUAL,
        action_encoding=ActionEncoding.JOINT_POS_BIMANUAL_NAV,
        data_dir=Path("~/tensorflow_datasets")
    ),

    "aloha_static_dataset": OXEDatasetConfig(
        robot_name="ViperX",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="cam_high", secondary="cam_low", wrist="cam_right_wrist"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT_BIMANUAL,
        action_encoding=ActionEncoding.JOINT_POS_BIMANUAL
    ),

    # RT1 and related
    "fractal20220817_data": OXEDatasetConfig(
        robot_name="RT1",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_QUAT,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # RoboTurk
    "roboturk": OXEDatasetConfig(
        robot_name="Sawyer",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="front_rgb"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.NONE,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # TACO Play
    "taco_play": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="rgb_static", wrist="rgb_gripper"),
        depth_obs_keys=ImageConfig(primary="depth_static", wrist="depth_gripper"),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # Jaco Play
    "jaco_play": OXEDatasetConfig(
        robot_name="Jaco 2",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="image_wrist"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    # Navigation datasets
    "gnm_dataset": OXEDatasetConfig(
        robot_name="Navigation Robot",
        action_space="2D waypoint",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_NAV,
        action_encoding=ActionEncoding.NAV_2D
    ),

    # LIBERO datasets
    "libero_spatial_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),

    "libero_object_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),

    "libero_goal_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),

    "libero_10_no_noops": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),

    # RoboSet
    "robo_set": OXEDatasetConfig(
        robot_name="Franka",
        action_space="absolute joint",
        image_obs_keys=ImageConfig(primary="image_top", secondary="image_left", wrist="image_wrist"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.JOINT_POS
    ),

    # Mixed Objects (MO)
    "MO": OXEDatasetConfig(
        robot_name="Mixed",
        action_space="mixed",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS,
        data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
    ),

    "nyu_door_opening_surprising_effectiveness": OXEDatasetConfig(
        robot_name="Hello Stretch",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(wrist="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.NONE,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "nyu_franka_play_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", secondary="image_additional_view"),
        depth_obs_keys=ImageConfig(primary="depth", secondary="depth_additional_view"),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "stanford_hydra_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "language_table": OXEDatasetConfig(
        robot_name="xArm",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="rgb"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "dlr_sara_pour_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="DLR SARA",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "cmu_playing_with_food": OXEDatasetConfig(
        robot_name="Franka",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", wrist="finger_vision_1"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="xArm",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image", secondary="image2", wrist="hand_image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.POS_EULER,
        action_encoding=ActionEncoding.EEF_POS
    ),

    "ucsd_kitchen_dataset_converted_externally_to_rlds": OXEDatasetConfig(
        robot_name="xArm",
        action_space="delta end-effector",
        image_obs_keys=ImageConfig(primary="image"),
        depth_obs_keys=ImageConfig(),
        proprio_encoding=ProprioEncoding.JOINT,
        action_encoding=ActionEncoding.EEF_POS
    ),
}