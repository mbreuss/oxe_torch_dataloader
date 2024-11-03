"""Configuration definitions for Open X-Embodiment datasets."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Set, Union

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
class DatasetConfig:
    """Core configuration for datasets."""
    image_obs_keys: ImageConfig
    depth_obs_keys: ImageConfig
    proprio_encoding: ProprioEncoding
    action_encoding: ActionEncoding
    data_dir: Optional[Path] = None
    language_key: Optional[str] = None
    dataset_size_limit: Optional[int] = None
    filter_functions: tuple = field(default_factory=tuple)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.data_dir and isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir).expanduser()


class DatasetRegistry:
    """Registry for dataset configurations."""
    
    def __init__(self):
        self._configs: Dict[str, DatasetConfig] = {}
        self._initialize_configs()

    def _initialize_configs(self):
        """Initialize all dataset configurations."""
        self._configs.update({
            # Bridge and variants
            "bridge_dataset": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image_0", secondary="image_1"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS,
                language_key="groundtruth*",
                filter_functions=(
                    ModuleSpec.create(filter_by_language_key, language_key_template="groundtruth*"),
                    ModuleSpec.create(filter_by_language_key, language_key_template="language_instruction*")
                )
            ),
            
            # Kitchen datasets
            "kit_irl_real_kitchen_lang": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image_top", secondary="image_side", wrist="wrist_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.JOINT,
                action_encoding=ActionEncoding.JOINT_POS,
                data_dir=Path("/home/marcelr/tensorflow_datasets"),
                language_key="language_instruction*"
            ),

            "kit_irl_real_kitchen_vis": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image_top", secondary="image_side", wrist=None),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.JOINT,
                action_encoding=ActionEncoding.JOINT_POS,
                data_dir=Path("/home/marcelr/tensorflow_datasets")
            ),

            # Droid dataset
            "droid": DatasetConfig(
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

            # Berkeley datasets
            "berkeley_mvp_converted_externally_to_rlds": DatasetConfig(
                image_obs_keys=ImageConfig(wrist="hand_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_QUAT,
                action_encoding=ActionEncoding.JOINT_POS,
                data_dir=Path("~/tensorflow_datasets")
            ),

            "berkeley_cable_routing": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", secondary=None, wrist="hand_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.JOINT,
                action_encoding=ActionEncoding.EEF_POS
            ),

            "berkeley_autolab_ur5": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", secondary=None, wrist="hand_image"),
                depth_obs_keys=ImageConfig(primary="depth"),
                proprio_encoding=ProprioEncoding.POS_QUAT,
                action_encoding=ActionEncoding.EEF_POS
            ),

            "berkeley_fanuc_manipulation": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", secondary=None, wrist="wrist_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.JOINT,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # RT1 and related
            "fractal20220817_data": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_QUAT,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # RoboTurk
            "roboturk": DatasetConfig(
                image_obs_keys=ImageConfig(primary="front_rgb"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.NONE,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # TACO Play
            "taco_play": DatasetConfig(
                image_obs_keys=ImageConfig(primary="rgb_static", wrist="rgb_gripper"),
                depth_obs_keys=ImageConfig(primary="depth_static", wrist="depth_gripper"),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # Jaco Play
            "jaco_play": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", wrist="image_wrist"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # ALOHA datasets
            "aloha_mobile": DatasetConfig(
                image_obs_keys=ImageConfig(primary="cam_high", wrist="cam_right_wrist"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.JOINT_BIMANUAL,
                action_encoding=ActionEncoding.JOINT_POS_BIMANUAL_NAV,
                data_dir=Path("~/tensorflow_datasets")
            ),

            "aloha_static_dataset": DatasetConfig(
                image_obs_keys=ImageConfig(
                    primary="cam_high",
                    secondary="cam_low",
                    wrist="cam_right_wrist"
                ),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.JOINT_BIMANUAL,
                action_encoding=ActionEncoding.JOINT_POS_BIMANUAL
            ),

            "aloha_dagger_dataset": DatasetConfig(
                image_obs_keys=ImageConfig(
                    primary="cam_high",
                    secondary="cam_low",
                    wrist="cam_right_wrist"
                ),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.JOINT_BIMANUAL,
                action_encoding=ActionEncoding.JOINT_POS_BIMANUAL
            ),

            # NYU datasets
            "nyu_door_opening_surprising_effectiveness": DatasetConfig(
                image_obs_keys=ImageConfig(wrist="image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.NONE,
                action_encoding=ActionEncoding.EEF_POS
            ),

            "nyu_franka_play_dataset_converted_externally_to_rlds": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", secondary="image_additional_view"),
                depth_obs_keys=ImageConfig(
                    primary="depth",
                    secondary="depth_additional_view"
                ),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # Stanford datasets
            "stanford_hydra_dataset_converted_externally_to_rlds": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # Language datasets
            "language_table": DatasetConfig(
                image_obs_keys=ImageConfig(primary="rgb"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # Navigation datasets
            "gnm_dataset": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_NAV,
                action_encoding=ActionEncoding.NAV_2D
            ),

            # RoboSet
            "robo_set": DatasetConfig(
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
            "ucsd_kitchen_dataset_converted_externally_to_rlds": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.JOINT,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # UT datasets
            "utokyo_xarm_pick_and_place_converted_externally_to_rlds": DatasetConfig(
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
            "dlr_sara_pour_converted_externally_to_rlds": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # CMU datasets
            "cmu_playing_with_food": DatasetConfig(
                image_obs_keys=ImageConfig(
                    primary="image",
                    wrist="finger_vision_1"
                ),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS
            ),

            # LIBERO datasets
            "libero_spatial_no_noops": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS,
                data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
            ),

            "libero_object_no_noops": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS,
                data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
            ),

            "libero_goal_no_noops": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS,
                data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
            ),

            "libero_10_no_noops": DatasetConfig(
                image_obs_keys=ImageConfig(primary="image", wrist="wrist_image"),
                depth_obs_keys=ImageConfig(),
                proprio_encoding=ProprioEncoding.POS_EULER,
                action_encoding=ActionEncoding.EEF_POS,
                data_dir=Path("/home/marcelr/tensorflow_datasets/modified_libero_rlds")
            ),

            # Additional datasets can be added here...
        })

    def get_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for a dataset."""
        if dataset_name not in self._configs:
            raise ValueError(f"No configuration found for dataset: {dataset_name}")
        return self._configs[dataset_name]

    def register_config(self, dataset_name: str, config: DatasetConfig):
        """Register a new dataset configuration."""
        self._configs[dataset_name] = config


# Create global registry instance
DATASET_REGISTRY = DatasetRegistry()

# Export configurations
OXE_DATASET_CONFIGS = {
    name: config 
    for name, config in DATASET_REGISTRY._configs.items()
}