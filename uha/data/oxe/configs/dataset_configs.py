"""Dataset configurations."""

from typing import Dict, Any
import logging

from pathlib import Path
from ..transforms.base import ImageKeyConfig

logger = logging.getLogger(__name__)

# Complete mapping of all dataset configurations
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Bridge and variants
    "bridge_dataset": {
        "transform_type": "franka",
        "robot_name": "WindowX",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image_0",
            "secondary": "image_1",
            "wrist": None
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos",
        "language_key": "groundtruth*",
    },

    # Kitchen datasets
    "kit_irl_real_kitchen_lang": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "absolute joint",
        "image_keys": {
            "primary": "image_top",
            "secondary": "image_side",
            "wrist": "wrist_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "joint",
        "action_encoding": "joint_pos",
        "data_dir": "/home/marcelr/tensorflow_datasets",
        "language_key": "language_instruction*"
    },

    "kit_irl_real_kitchen_vis": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "absolute joint",
        "image_keys": {
            "primary": "image_top",
            "secondary": "image_side",
            "wrist": None
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "joint",
        "action_encoding": "joint_pos",
        "data_dir": "/home/marcelr/tensorflow_datasets"
    },

    # Droid dataset
    "droid": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "absolute joint",
        "image_keys": {
            "primary": "exterior_image_1_left",
            "secondary": "exterior_image_2_left",
            "wrist": "wrist_image_left"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "joint",
        "action_encoding": "joint_pos",
        "language_key": "language_instruction*",
        "dataset_size_limit": 1000
    },

    # Berkeley datasets
    "berkeley_mvp_converted_externally_to_rlds": {
        "transform_type": "franka",
        "robot_name": "xArm",
        "action_space": "absolute joint",
        "image_keys": {
            "primary": None,
            "secondary": None,
            "wrist": "hand_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_quat",
        "action_encoding": "joint_pos",
        "data_dir": "~/tensorflow_datasets"
    },

    "berkeley_cable_routing": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "hand_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "joint",
        "action_encoding": "eef_pos"
    },

    "berkeley_autolab_ur5": {
        "transform_type": "ur5",
        "robot_name": "UR5",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "hand_image"
        },
        "depth_keys": {
            "primary": "depth",
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_quat",
        "action_encoding": "eef_pos"
    },

    # ALOHA datasets
    "aloha_mobile": {
        "transform_type": "bimanual",
        "robot_name": "ViperX",
        "action_space": "absolute joint",
        "num_arms": 2,
        "image_keys": {
            "primary": "cam_high",
            "secondary": None,
            "wrist": "cam_right_wrist"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "joint_bimanual",
        "action_encoding": "joint_pos_bimanual_nav",
        "data_dir": "~/tensorflow_datasets"
    },

    "aloha_static_dataset": {
        "transform_type": "bimanual",
        "robot_name": "ViperX",
        "action_space": "absolute joint",
        "num_arms": 2,
        "image_keys": {
            "primary": "cam_high",
            "secondary": "cam_low",
            "wrist": "cam_right_wrist"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "joint_bimanual",
        "action_encoding": "joint_pos_bimanual"
    },

    # LIBERO datasets
    "libero_spatial_no_noops": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos",
        "data_dir": "/home/marcelr/tensorflow_datasets/modified_libero_rlds"
    },

    "libero_object_no_noops": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos",
        "data_dir": "/home/marcelr/tensorflow_datasets/modified_libero_rlds"
    },
    # RT1 and related
    "fractal20220817_data": {
        "transform_type": "franka",
        "robot_name": "RT1",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": None
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_quat",
        "action_encoding": "eef_pos"
    },

    # RoboTurk
    "roboturk": {
        "transform_type": "franka",
        "robot_name": "Sawyer",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "front_rgb",
            "secondary": None,
            "wrist": None
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "none",
        "action_encoding": "eef_pos"
    },

    # TACO Play
    "taco_play": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "rgb_static",
            "secondary": None,
            "wrist": "rgb_gripper"
        },
        "depth_keys": {
            "primary": "depth_static",
            "secondary": None,
            "wrist": "depth_gripper"
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos"
    },

    # Jaco Play
    "jaco_play": {
        "transform_type": "franka",
        "robot_name": "Jaco 2",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "image_wrist"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos"
    },

    # Navigation datasets
    "gnm_dataset": {
        "transform_type": "franka",
        "robot_name": "Navigation Robot",
        "action_space": "2D waypoint",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": None
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_nav",
        "action_encoding": "nav_2d"
    },

    # RoboSet
    "robo_set": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "absolute joint",
        "image_keys": {
            "primary": "image_top",
            "secondary": "image_left",
            "wrist": "image_wrist"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "joint",
        "action_encoding": "joint_pos"
    },

    # UCSD datasets
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "transform_type": "franka",
        "robot_name": "xArm",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": None
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "joint",
        "action_encoding": "eef_pos"
    },

    # UT datasets
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "transform_type": "franka",
        "robot_name": "xArm",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": "image2",
            "wrist": "hand_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos"
    },

    # DLR datasets
    "dlr_sara_pour_converted_externally_to_rlds": {
        "transform_type": "franka",
        "robot_name": "DLR SARA",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": None
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos"
    },

    # CMU datasets
    "cmu_playing_with_food": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "finger_vision_1"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos"
    },

    # Language datasets
    "language_table": {
        "transform_type": "franka",
        "robot_name": "xArm",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "rgb",
            "secondary": None,
            "wrist": None
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos"
    },

    # NYU datasets
    "nyu_door_opening_surprising_effectiveness": {
        "transform_type": "franka",
        "robot_name": "Hello Stretch",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": None,
            "secondary": None,
            "wrist": "image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "none",
        "action_encoding": "eef_pos"
    },

    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": "image_additional_view",
            "wrist": None
        },
        "depth_keys": {
            "primary": "depth",
            "secondary": "depth_additional_view",
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos"
    },

    # Stanford datasets
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos"
    },
    "libero_goal_no_noops": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos",
        "data_dir": "/home/marcelr/tensorflow_datasets/modified_libero_rlds"
    },

    "libero_10_no_noops": {
        "transform_type": "franka",
        "robot_name": "Franka",
        "action_space": "delta end-effector",
        "image_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image"
        },
        "depth_keys": {
            "primary": None,
            "secondary": None,
            "wrist": None
        },
        "proprio_encoding": "pos_euler",
        "action_encoding": "eef_pos",
        "data_dir": "/home/marcelr/tensorflow_datasets/modified_libero_rlds"
    },
}