"""Open X-Embodiment Dataset Transforms

input: dict of features, each is batched, i.e. has leading time dimension
expected output:
step = {
    'observation': {
        <image_keys, depth_image_keys>
        state in chosen state representation
    },
    'action': action in chosen action representation,
    'language_instruction': str,
}
"""

from typing import Any, Dict

import tensorflow as tf

from uha.data.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_actions,
)

def add_robot_information(robot_name, action_space, number_arms):
    if(number_arms > 1):
        return "A {robot_name} robot with {number_arms} arms controlled by {action_space} actions".format(robot_name=robot_name, number_arms=number_arms, action_space=action_space)
    else:
        return "A {robot_name} robot with {number_arms} arm controlled by {action_space} actions".format(robot_name=robot_name, number_arms=number_arms, action_space=action_space)


def bridge_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # marcel's?
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    # TODO: confirm we need this for marcel's
    trajectory = relabel_actions(trajectory)

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("WindowX", "delta end-effector", 1)
    return trajectory


def kit_irl_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1], 0.05, 0.01)[:, None],
        ],
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pos"][:, :],
            trajectory["observation"]["end_effector_ori"][:, :],
            binarize_gripper_actions(trajectory["action_abs"][:, -1], 0.05, 0.01)[:, None],
        ),
        axis=1
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def kit_irl_dataset_joint_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["delta_des_joint_state"][:, :7],
            binarize_gripper_actions(trajectory["action"][:, -1], 0.05, 0.01)[:, None],
        ],
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_state"][:, :],
            binarize_gripper_actions(trajectory["action_abs"][:, -1], 0.05, 0.01)[:, None],
        ),
        axis=1
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta joint", 1)
    return trajectory


def kit_irl_dataset_abs_joint_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action_joint_state"][:, :7],
            binarize_gripper_actions(trajectory["action"][:, -1], 0.05, 0.01)[:, None],
        ],
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_state"][:, :],
            binarize_gripper_actions(trajectory["action_abs"][:, -1], 0.05, 0.01)[:, None],
        ),
        axis=1
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    return trajectory


def kit_irl_dataset_abs_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action_abs"][:, :6],
            binarize_gripper_actions(trajectory["action_abs"][:, -1], 0.05, 0.01)[:, None],
        ],
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pos"][:, :],
            trajectory["observation"]["end_effector_ori"][:, :],
            binarize_gripper_actions(trajectory["action_abs"][:, -1], 0.05, 0.01)[:, None],
        ),
        axis=1
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "absolute end-effector", 1)
    return trajectory


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action_dict"]["joint_position"][:, :7],
            binarize_gripper_actions(trajectory["action_dict"]["gripper_position"][:, -1], 0.95, 0.05)[:, None],
        ],
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_position"][:, :],
            binarize_gripper_actions(trajectory["observation"]["gripper_position"][:, -1], 0.95, 0.05)[:, None],
        ),
        axis=1
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    return trajectory


def bridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # NOTE: this is not actually the official OXE copy of bridge, it is our own more up-to-date copy that you
    # can find at https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1], 0.95, 0.05)[:, None],
        ],
        axis=-1,
    )
    trajectory = relabel_actions(trajectory)
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("WindowX", "delta end-effector", 1)
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"],
            trajectory["observation"]["gripper_closed"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    trajectory["observation"]["robot_information"] = add_robot_information("WindowX", "delta end-effector", 1)
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    gripper_value = tf.io.decode_compressed(
        trajectory["observation"]["gripper_closed"], compression_type="ZLIB"
    )
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["proprio"] = tf.concat(
        (
            tf.reshape(eef_value, (-1, 7)),
            tf.reshape(gripper_value, (-1, 1)),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Kuka iiwa", "delta end-effector", 1)
    return trajectory


def taco_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["robot_obs"][:, :6],
            trajectory["observation"]["robot_obs"][:, 7:8],
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "end_effector_cartesian_pos"
    ]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    trajectory["observation"]["robot_information"] = add_robot_information("Jaco 2", "delta end-effector", 1)
    return trajectory


def berkeley_cable_routing_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(
        tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1)
    )

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Sawyer", "delta end-effector", 1)
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Hello Stretch", "delta end-effector", 1)
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_states"],
            trajectory["observation"]["gripper_states"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def berkeley_autolab_ur5_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = trajectory["observation"].pop(
        "image_with_depth"
    )

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"][
        :, 6:14
    ]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    trajectory["observation"]["robot_information"] = add_robot_information("UR5", "delta end-effector", 1)
    return trajectory


def toto_dataset_transform_eef(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def toto_dataset_transform_joint(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    joint_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :],
            tf.cast(trajectory["action"]["open_gripper"][:-1, None], tf.float32),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = joint_actions
    traj_truncated["observation"]["proprio"] = traj_truncated["observation"]["state"]
    traj_truncated["language_instruction"] = tf.fill(
        tf.shape(traj_truncated["observation"]["natural_language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    return traj_truncated


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "effector_translation"
    ]
    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(
        instruction_bytes, output_encoding="UTF-8"
    )
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[
        :, :1
    ].to_tensor()[:, 0]
    trajectory["observation"]["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["robot_state"]
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    trajectory["observation"]["robot_information"] = add_robot_information("UR5", "delta end-effector", 1)
    return trajectory


def stanford_kuka_multimodal_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][
        ..., 0
    ]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["ee_position"],
            trajectory["observation"]["ee_orientation"],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Kuka iiwa", "delta end-effector", 1)
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :7]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    return trajectory


def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
            trajectory["observation"]["state"][:, -3:-2],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :8]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(
        trajectory["observation"]["depth"][..., 0], tf.float32
    )
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, -6:]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_pose"],
            trajectory["observation"]["state"][:, 7:8],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def cmu_franka_exploration_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(
                tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["present/xyz"],
            trajectory["observation"]["present/axis_angle"],
            trajectory["observation"]["present/sensed_close"],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    trajectory["observation"]["robot_information"] = add_robot_information("Google Robot", "delta end-effector", 1)
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("PR2", "delta end-effector", 1)
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("PR2", "delta end-effector", 1)
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["observation"]["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "end_effector_pose"
    ]
    trajectory["observation"]["robot_information"] = add_robot_information("xArm", "delta end-effector", 2)
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
            trajectory["observation"]["state"][:, -1:],
        ),
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Multi-Robot", "delta end-effector", 1)
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["pose"],
            tf.cast(trajectory["observation"]["gripper"], tf.float32)[:, None],
        ),
        axis=-1,
    )

    # invert gripper
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :-1] + trajectory["observation"]["joint_pos"],
            # trajectory["action"][:, :-1],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ],
        axis=1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("xArm", "absolute joint", 1)

    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 30Hz to 10Hz
    factor = 3
    trajectory = tf.nest.map_structure(lambda x: x[::factor], trajectory)

    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_pos"],
            tf.cast(trajectory["observation"]["gripper"], tf.float32)[:, None],
        ),
        axis=-1,
    )

    # recompute actions for downsampled sequence
    joint_actions = (
        trajectory["observation"]["joint_pos"][1:, :7]
        # - trajectory["observation"]["joint_pos"][:-1, :7]
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)

    # recombine to get full actions, invert gripper
    traj_truncated["action"] = tf.concat(
        [joint_actions, invert_gripper_actions(trajectory["action"][:-1, -1:])],
        axis=1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "absolute joint", 1)

    return traj_truncated


def kaist_nonprehensible_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, -7:]
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
            trajectory["observation"]["end_effector_pose"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Sawyer", "delta end-effector", 1)
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("DLR SARA", "delta end-effector", 1)
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["robot_information"] = add_robot_information("DLR SARA", "delta end-effector", 1)
    return trajectory


def dlr_edan_shared_control_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("DLR EDAN", "delta end-effector", 1)
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["ground_truth_states"]["EE"],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("UR5", "delta end-effector", 1)
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Sawyer", "delta end-effector", 1)
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, 7:8],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Kinova Gen3", "delta end-effector", 1)
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(
                tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)
            ),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :8]
    trajectory["language_instruction"] = tf.fill(
        tf.shape(trajectory["language_instruction"]), ""
    )  # delete uninformative language instruction
    trajectory["observation"]["robot_information"] = add_robot_information("PAMY2", "delta end-effector", 1)
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            trajectory["observation"]["state"][:, 6:7],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Fanuc Mate", "delta end-effector", 1)
    return trajectory


def cmu_playing_with_food_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Hello Stretch", "delta end-effector", 1)
    return trajectory


def gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    def subsampled_traj():
        # first compute per-dataset scaling factor from first action and first 2 positions
        scaling_factor = tf.linalg.norm(trajectory["action"][0]) / tf.linalg.norm(
            trajectory["observation"]["position"][1]
            - trajectory["observation"]["position"][0]
        )
        # subsample trajectory by factor of 3
        subsample_factor = 3
        traj = tf.nest.map_structure(lambda x: x[::subsample_factor], trajectory)
        # recompute actions from position and yaw
        yaw = traj["observation"]["yaw"]
        pos = traj["observation"]["position"]
        rot_mat = tf.convert_to_tensor(
            [
                [tf.cos(yaw), -tf.sin(yaw)],
                [tf.sin(yaw), tf.cos(yaw)],
            ]
        )
        rot_mat = tf.transpose(rot_mat, [3, 2, 0, 1])[0]
        delta = pos[1:] - pos[:-1]
        action = tf.matmul(delta[:, None], rot_mat[:-1])[:, 0] * scaling_factor
        # truncate last element for all other keys
        traj = tf.nest.map_structure(lambda x: x[:-1], traj)
        traj["action"] = action
        return traj

    def dummy_traj():
        return tf.nest.map_structure(lambda x: x[:0], trajectory)

    # we need to filter out trajectories of length 1 in order to compute the scaling factor
    trajectory = tf.cond(
        tf.shape(trajectory["action"])[0] > 1, subsampled_traj, dummy_traj
    )

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    return trajectory


def aloha_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 50Hz to 10Hz
    factor = 5
    trajectory = tf.nest.map_structure(lambda x: x[::factor], trajectory)

    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("ViperX", "absolute joint", 2)
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"],
            trajectory["observation"]["state_gripper_pose"][..., None],
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["observation"]["robot_information"] = add_robot_information("Hello Stretch", "delta end-effector", 1)
    return trajectory


def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7] + trajectory["observation"]["state"][:, :7],
            # trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def mujoco_manip_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    gripper_action = invert_gripper_actions(trajectory["action"][:, -1:] / 255)
    trajectory["action"] = tf.concat(
        (trajectory["action"][:, :6], gripper_action), axis=-1
    )
    return trajectory


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    trajectory["observation"]["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge": bridge_transform,
    "bridge_dataset_without_single": bridge_transform,
    "kit_irl_real_kitchen_delta_des_joint_euler": kit_irl_dataset_abs_joint_transform, # old
    "kit_irl_real_kitchen_vis_delta_des_joint_euler": kit_irl_dataset_abs_joint_transform, # old
    "kit_irl_real_kitchen_vis": kit_irl_dataset_abs_joint_transform, # kit_irl_dataset_abs_joint_transform # kit_irl_dataset_joint_transform # kit_irl_dataset_abs_transform # kit_irl_dataset_transform
    "kit_irl_real_kitchen_lang": kit_irl_dataset_abs_joint_transform, # kit_irl_dataset_abs_joint_transform # kit_irl_dataset_joint_transform # kit_irl_dataset_abs_transform # kit_irl_dataset_transform
    "droid": droid_dataset_transform,
    "bridge_dataset": bridge_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform_joint,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "gnm_dataset": gnm_dataset_transform,
    "aloha_static_dataset": aloha_dataset_transform,
    "aloha_dagger_dataset": aloha_dataset_transform,
    "aloha_mobile": aloha_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "robo_set": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    "mujoco_manip": mujoco_manip_dataset_transform,
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
}