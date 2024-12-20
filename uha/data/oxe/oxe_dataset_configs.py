"""Dataset kwargs for Open X-Embodiment datasets.

Target configuration:
    image_obs_keys:
        primary: primary external RGB
        secondary: secondary external RGB
        wrist: wrist RGB
    depth_obs_keys:
        primary: primary external depth
        secondary: secondary external depth
        wrist: wrist depth
    proprio_encoding: Type of proprio encoding used
    action_encoding: Type of action encoding used, e.g. EEF position vs joint position control
"""
from enum import IntEnum

from uha.data.utils.data_utils import filter_by_language_key, filter_by_task_and_language_key
from uha.data.utils.spec import ModuleSpec
from .transforms.droid_utils import zero_action_filter


pnp_task_list_bridge  =  ["put", "pick", "take", "pnp", "icra", "rss", "many_skills", "lift_bowl", "move_drying", "wipe", "right_pepper", "topple", "upright", "flip"]


class ProprioEncoding(IntEnum):
    """Defines supported proprio encoding schemes for different datasets."""

    NONE = -1  # no proprio provided
    POS_EULER = 1  # EEF XYZ + roll-pitch-yaw + gripper open/close
    POS_QUAT = 2  # EEF XYZ + quaternion + gripper open/close
    JOINT = 3  # joint angles + gripper open/close
    JOINT_BIMANUAL = 4  # 2 x [6 x joint angles + gripper open/close]
    POS_NAV = 5  # XY + yaw


class ActionEncoding(IntEnum):
    """Defines supported action encoding schemes for different datasets."""

    EEF_POS = 1  # EEF delta XYZ + roll-pitch-yaw + gripper open/close
    JOINT_POS = 2  # 7 x joint delta position + gripper open/close
    JOINT_POS_BIMANUAL = 3  # 2 x [6 x joint pos + gripper]
    NAV_2D = 4  # [delta_x, delta_y] waypoint
    JOINT_POS_BIMANUAL_NAV = (
        5  # 2 x [6 x joint pos + gripper] + linear base vel + angular base vel
    )


OXE_DATASET_CONFIGS = {
    "bridge": {
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        # for groundtruth
        "language_key": "groundtruth*",
        # for lupus
        # "language_key": "language_instruction*",

        # take the intersection of both labels as our dataset
        "filter_functions": (ModuleSpec.create(
            filter_by_language_key,
            language_key_template="groundtruth*"
        ), ModuleSpec.create(
            filter_by_language_key,
            language_key_template="language_instruction*"
        ))
    },
    # old, use below
    "kit_irl_real_kitchen_delta_des_joint_euler": {
        "image_obs_keys": {"primary": "image", "secondary": "wrist_image", "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "data_dir": "~/tensorflow_datasets",
        # "data_dir": "/hkfs/work/workspace/scratch/unesl-datasets",
        "language_key": "language_instruction*",
        # "shuffle": False,
    },
    # old, use below
    "kit_irl_real_kitchen_vis_delta_des_joint_euler": {
        "image_obs_keys": {"primary": "image_top", "secondary": "image_side", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "data_dir": "~/tensorflow_datasets",
        # "data_dir": "/hkfs/work/workspace/scratch/unesl-datasets",
        # "shuffle": False,
    },
    "kit_irl_real_kitchen_lang": {
        "image_obs_keys": {"primary": "image_top", "secondary": "image_side", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "data_dir": "/home/marcelr/tensorflow_datasets",
        "language_key": "language_instruction*",
        # "shuffle": False,
    },
    "kit_irl_real_kitchen_vis": {
        "image_obs_keys": {"primary": "image_top", "secondary": "image_side", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "data_dir": "/home/marcelr/tensorflow_datasets",
        # "shuffle": False,
    },
    "droid": {
        "image_obs_keys": {
            "primary": "exterior_image_1_left", 
            "secondary": None, 
            "wrist": "wrist_image_left"
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "language_key": "language_instruction*",
        # "dataset_size_limit": 1000,
        # "shuffle": False,
    },
    "eef_droid": {
        "image_obs_keys": {
            "primary": "exterior_image_1_left", 
            # "secondary": "exterior_image_2_left", 
            "secondary": None,
            "wrist": "wrist_image_left"
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "language_key": "language_instruction*",
       #  "dataset_size_limit": 1000,
        # "shuffle": False,
    },
    "fractal20220817_data": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "kuka": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
        "language_key": "natural_language_instruction",  # No wildcard since we're using this exact key
    },
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
    "bridge_dataset": {
        "image_obs_keys": {"primary": "image_0", "secondary": "image_1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        # "data_dir": "~/tensorflow_datasets",
        "data_dir": "/home/marcelr/tensorflow_datasets",
        # "data_dir": "/hkfs/work/workspace/scratch/unesl-datasets/rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds",
    },
    "taco_play": {
        "image_obs_keys": {
            "primary": "rgb_static",
            "secondary": None,
            "wrist": "rgb_gripper",
        },
        "depth_obs_keys": {
            "primary": "depth_static",
            "secondary": None,
            "wrist": "depth_gripper",
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "jaco_play": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_cable_routing": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "top_image",
            "wrist": "wrist45_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
        "language_key": "language_instruction*",  # Add wildcard
    },
    "roboturk": {
        "image_obs_keys": {"primary": "front_rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_door_opening_surprising_effectiveness": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "viola": {
        "image_obs_keys": {
            "primary": "agentview_rgb",
            "secondary": None,
            "wrist": "eye_in_hand_rgb",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_autolab_ur5": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "hand_image",
        },
        "depth_obs_keys": {"primary": "depth", "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "toto": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "data_dir": "~/tensorflow_datasets",
        # "data_dir": "/hkfs/work/workspace/scratch/unesl-datasets",
    },
    "language_table": {
        "image_obs_keys": {"primary": "rgb", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "columbia_cairlab_pusht_real": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": "depth_image", "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_rot_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_buds_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image_additional_view",
            "wrist": None,
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": "depth_additional_view",
            "wrist": None,
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "maniskill_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {
            "primary": "depth",
            "secondary": None,
            "wrist": "wrist_depth",
        },
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "furniture_bench_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "highres_image",
            "secondary": None,
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "bc_z": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "image2",
            "wrist": "hand_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utokyo_xarm_bimanual_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "robo_net": {
        "image_obs_keys": {"primary": "image", "secondary": "image1", "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_mvp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "data_dir": "~/tensorflow_datasets",
        # "data_dir": "/hkfs/work/workspace/scratch/unesl-datasets",
    },
    "berkeley_rpt_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "hand_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        "data_dir": "~/tensorflow_datasets",
        # "data_dir": "/hkfs/work/workspace/scratch/unesl-datasets",
    },
    "kaist_nonprehensile_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_QUAT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_mask_vit_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "tokyo_u_lsmo_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_pour_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_sara_grid_clamp_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "asu_table_top_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "stanford_robocook_converted_externally_to_rlds": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "imperialcollege_sawyer_wrist_cam": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "uiuc_d3field": {
        "image_obs_keys": {"primary": "image_1", "secondary": "image_2", "wrist": None},
        "depth_obs_keys": {"primary": "depth_1", "secondary": "depth_2", "wrist": None},
        "proprio_encoding": ProprioEncoding.NONE,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "utaustin_mutex": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "berkeley_fanuc_manipulation": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_playing_with_food": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": "finger_vision_1",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_play_fusion": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "cmu_stretch": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "gnm_dataset": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_NAV,
        "action_encoding": ActionEncoding.NAV_2D,
    },
    "aloha_static_dataset": {
        "image_obs_keys": {
            "primary": "cam_high",
            "secondary": "cam_low",
            "wrist": "cam_right_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_dagger_dataset": {
        "image_obs_keys": {
            "primary": "cam_high",
            "secondary": "cam_low",
            "wrist": "cam_right_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
    },
    "aloha_mobile": {
        "image_obs_keys": {
            "primary": "cam_high",
            "secondary": None,
            "wrist": "cam_right_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL_NAV,
        # "data_dir": "~/tensorflow_datasets",
        # "data_dir": "/hkfs/work/workspace/scratch/unesl-datasets",
    },
    "fmb_dataset": {
        "image_obs_keys": {
            "primary": "image_side_1",
            "secondary": "image_side_2",
            "wrist": "image_wrist_1",
        },
        "depth_obs_keys": {
            "primary": "image_side_1_depth",
            "secondary": "image_side_2_depth",
            "wrist": "image_wrist_1_depth",
        },
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "dobbe": {
        "image_obs_keys": {"primary": None, "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "robo_set": {
        "image_obs_keys": {
            "primary": "image_top",
            "secondary": "image_left",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT,
        "action_encoding": ActionEncoding.JOINT_POS,
        # "dataset_size_limit": 2000,
    },
    "rh20t": {
        "image_obs_keys": {
            "primary": "image_front",
            "secondary": "image_side_right",
            "wrist": "image_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "mujoco_manip": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": None,
            "wrist": None,
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
    },
    "aloha_play_dataset": {
        "image_obs_keys": {
            "primary": "cam_high",
            "secondary": "cam_low",
            "wrist": "cam_right_wrist",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.JOINT_BIMANUAL,
        "action_encoding": ActionEncoding.JOINT_POS_BIMANUAL,
        "data_dir": "/home/reuss/.cache/huggingface/hub/datasets--oier-mees--BiPlay/snapshots/6af6c5ca72bbe40b0a3db80adca041b6adf526c1/BiPlay", # aloha_play_dataset/1.0.0
    },
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        # "data_dir": "~/tensorflow_datasets/modified_libero_rlds",
        "data_dir": "/home/marcelr/tensorflow_datasets/modified_libero_rlds",
    },
    "libero_object_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        # "data_dir": "~/tensorflow_datasets/modified_libero_rlds",
        "data_dir": "/home/marcelr/tensorflow_datasets/modified_libero_rlds",
    },
    "libero_goal_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        # "data_dir": "~/tensorflow_datasets/modified_libero_rlds",
        "data_dir": "/home/marcelr/tensorflow_datasets/modified_libero_rlds",
    },
    "libero_10_no_noops": {
        "image_obs_keys": {"primary": "image", "secondary": None, "wrist": "wrist_image"},
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "proprio_encoding": ProprioEncoding.POS_EULER,
        "action_encoding": ActionEncoding.EEF_POS,
        # "data_dir": "~/tensorflow_datasets/modified_libero_rlds",
        "data_dir": "/home/marcelr/tensorflow_datasets/modified_libero_rlds",
    },
}