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

def clean_instruction(input_str):
   input_str = str(input_str)
   input_str = input_str.split("tf.Tensor(b'")[-1].split("'")[0]
   return input_str

import tensorflow as tf

def format_instruction(
    instruction: tf.Tensor,
    robot_name: str,
    number_arms: str,
    action_space: str,
    prompt_style: str = "combined"
) -> tf.Tensor:
    """
    Wrapper to convert TF tensor instructions into formatted Florence prompts.

    Args:
        instruction (tf.Tensor): Tensor of instructions to format.
        robot_name (str): Name of the robot.
        number_arms (str): Number of robot arms.
        action_space (str): Action space type.
        prompt_style (str): Style of the prompt. Default is "combined".

    Returns:
        tf.Tensor: Tensor of formatted instructions.
    """

    def format_single_instruction(instr):
        # Decode the input tensor into a UTF-8 string
        if tf.is_tensor(instr):
            instr_str = tf.strings.as_string(instr) if instr.dtype != tf.string else instr
        else:
            instr_str = tf.convert_to_tensor(str(instr), dtype=tf.string)

        # Apply formatting using generate_policy_prompt
        def format_python_string(x):
            # Decode the instruction string
            decoded_instr = x.decode("utf-8") if isinstance(x, bytes) else str(x)
            # print(decoded_instr)
            # Format the prompt
            formatted_prompt = generate_policy_prompt(
                instruction=clean_instruction(decoded_instr),
                robot_name=robot_name,
                num_arms=int(number_arms),
                action_space=action_space,
                prompt_style=prompt_style
            )
            
            # Replace the tf.Tensor wrapper with the decoded instruction
            formatted_prompt = formatted_prompt.replace(
                f"Task Instruction: {x}",
                f"Task Instruction: {decoded_instr}"
            )
            
            return formatted_prompt

        # Use tf.py_function to apply the formatting function
        formatted = tf.py_function(
            func=format_python_string,
            inp=[instr_str],
            Tout=tf.string
        )

        return formatted

    # Map the formatting function to all elements in the instruction tensor
    return tf.map_fn(
        format_single_instruction,
        instruction,
        dtype=tf.string
    )



def generate_policy_prompt(
    instruction: str,
    robot_name: str = "UR5",
    num_arms: int = 1,
    action_space: str = "7D continuous",
    prompt_style: str = "combined",
    include_meta: bool = True
) -> str:
    """
    Generate structured prompts for VLA policy training.
    
    Args:
        instruction: Task instruction text
        robot_name: Name of the robot
        num_arms: Number of robot arms
        action_space: Description of action space
        prompt_style: Prompt generation strategy ("combined", "structured", "visual", "minimal")
        include_meta: Whether to include metadata tags
    
    Returns:
        Formatted prompt string
    """
    if isinstance(instruction, tf.Tensor):
        try:
            instruction = instruction.numpy().decode("utf-8")
        except AttributeError:
            raise ValueError(
                "The provided 'instruction' is a symbolic TensorFlow tensor. Ensure it is resolved to a string before passing."
            )
    # Base metadata string
    meta_info = f"Agent Type: {num_arms}-arm {robot_name}, Action Space: {action_space}, "
    
    prompts = {
        # Combines structured info with visual grounding
        "combined": f"""
            {meta_info if include_meta else ''}
            </od>Task Instruction: {instruction}</od><grounding>identify objects and spatial relationships for robotic manipulation</grounding>
        """,
        
        # Focuses on visual and spatial features
        "visual": f"""
            <od>Task Instruction: {instruction}, </od>
            <grounding>identify key objects and their spatial relationships</grounding>
            <region_cap>analyze motion paths and collision-free trajectories</region_cap>
            <dense_region_caption>determine optimal grasp points and manipulation targets</dense_region_caption>
            {f'<cap>{meta_info}</cap>' if include_meta else ''}
        """,
        
        # Structured format with clear sections
        "structured": f"""
            <od>ROBOT CONFIGURATION:
            {meta_info if include_meta else ''}
            
            TASK OBJECTIVE:
            {instruction}
            
            ANALYSIS REQUIREMENTS:
            - Identify target objects and obstacles
            - Determine spatial relationships
            - Plan manipulation sequence</od>
        """,
        
        # Minimal prompt for simpler tasks
        "minimal": f"""
            <od>{instruction}</od>
            <grounding>analyze for robotic manipulation</grounding>
            {f'<cap>{meta_info}</cap>' if include_meta else ''}
        """
    }
    
    if prompt_style not in prompts:
        raise ValueError(f"Invalid prompt style: {prompt_style}. Choose from: {list(prompts.keys())}")
    
    # Clean up whitespace and formatting
    prompt = prompts[prompt_style].strip()
    prompt = ' '.join(line.strip() for line in prompt.split('\n'))
    return prompt




'''def add_robot_information(robot_name, action_space, number_arms):
    if(number_arms > 1):
        return "A {robot_name} robot with {number_arms} arms controlled by {action_space} actions".format(robot_name=robot_name, number_arms=number_arms, action_space=action_space)
    else:
        return "A {robot_name} robot with {number_arms} arm controlled by {action_space} actions".format(robot_name=robot_name, number_arms=number_arms, action_space=action_space)'''


def add_robot_information(robot_name, action_space, number_arms):
    if number_arms > 1:
        info = "A {robot_name} robot with {number_arms} arms controlled by {action_space} actions".format(
            robot_name=robot_name, number_arms=number_arms, action_space=action_space)
    else:
        info = "A {robot_name} robot with {number_arms} arm controlled by {action_space} actions".format(
            robot_name=robot_name, number_arms=number_arms, action_space=action_space)
    
    # Convert the string to a TensorFlow tensor of type tf.string
    info_tensor = tf.convert_to_tensor(info, dtype=tf.string)
    return tf.reshape(info_tensor, [-1])


def get_action_space_index(robot_type, num_arms, control_mode='position', return_tensor=True):
    # Validate num_arms input
    if num_arms not in [1, 2]:
        raise ValueError("num_arms must be either 1 or 2")

    # Mapping of (robot_type, control_mode, num_arms) to indices
    action_space_mapping = {
        ('JOINT_POS', 'position', 1): 0,  # end-effector pos-1-arm pos
        # ('EEF_POS', 'velocity', 1): 1,  # end-effector delta-1-arm
        # ('JOINT_POS', 'position', 1): 2,  # joint-1-arm pos
        # ('EEF_POS', 'position', 2): 3,  # end-effector pos-2-arm pos
        ('EEF_POS', 'velocity', 1): 1,  # end-effector delta-2-arm
        ('JOINT_POS_BIMANUAL_NAV', 'position', 2): 2,  # joint-2-arm pos with navigation
        ('JOINT_POS_BIMANUAL', 'position', 2): 2,  # joint-2-arm pos (unified for bimanual or regular)
        ('JOINT_POS_NAV', 'position', 1): 0,  # joint-1-arm pos with navigation 
        ('EEF_POS_NAV', 'velocity', 1): 1,  # end-effector delta-2-arm
    }
    
    # Get the index from the mapping
    index = action_space_mapping.get((robot_type, control_mode, num_arms))
    
    if index is None:
        raise ValueError(f"Unsupported combination of robot_type: {robot_type}, control_mode: {control_mode}, and num_arms: {num_arms}")
    if return_tensor:
        # Convert to TensorFlow tensor
        return tf.constant(index, dtype=tf.int32)
    else:
        return index



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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta joint", 1)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS', 1, 'position')

    trajectory["language_instruction"] = format_instruction(
        trajectory["language_instruction"],
        robot_name="Franka Panda",
        action_space="joint position",
        number_arms="1",
        prompt_style='combined'
    )
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "absolute end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'position')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


def droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action_dict"]["joint_position"][:, :7],
            binarize_gripper_actions(trajectory["action_dict"]["gripper_position"][:, -1], 0.95, 0.05)[:, None],
        ],
        axis=-1,
    )
    trajectory['language_instruction'] = format_instruction(
        trajectory['language_instruction'],
        "Franka",
        "1",
        "joint position",
        prompt_style="combined"
    )
    trajectory["language_instruction_2"] = format_instruction(
        trajectory['language_instruction_2'],
        "Franka Panda",
        "1",
        "joint position",
        prompt_style="combined"
    )
    trajectory["language_instruction_3"] = format_instruction(
        trajectory['language_instruction_3'],
        "Franka Panda",
        "1",
        "joint position",
        prompt_style="combined"
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["joint_position"][:, :],
            binarize_gripper_actions(trajectory["observation"]["gripper_position"][:, -1], 0.95, 0.05)[:, None],
        ),
        axis=1
    )
    trajectory["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    # trajectory['frequency'] = tf.constant(15, dtype=tf.int32)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS', 1, 'position')
    return trajectory


def eef_droid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory['language_instruction'] = format_instruction(
        trajectory['language_instruction'],
        "Franka",
        "1",
        "Delta End-Effector",
        prompt_style="combined"
    )
    trajectory["language_instruction_2"] = format_instruction(
        trajectory['language_instruction_2'],
        "Franka Panda",
        "1",
        "Delta End-Effector",
        prompt_style="combined"
    )
    trajectory["language_instruction_3"] = format_instruction(
        trajectory['language_instruction_3'],
        "Franka Panda",
        "1",
        "Delta End-Effector",
        prompt_style="combined"
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["cartesian_position"][:, :6],
            binarize_gripper_actions(trajectory["observation"]["gripper_position"][:, -1], 0.95, 0.05)[:, None],
        ),
        axis=1
    )
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(15, dtype=tf.int32)
    return trajectory


def bridge_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # marcel's?
    temp_dict ={}
    temp_dict['action'] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    # TODO: confirm we need this for marcel's
    temp_dict = relabel_actions(temp_dict)

    # now move to unifed action space
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("WindowX", "delta end-effector", 1)

    trajectory['action'] = temp_dict['action']
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
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
    trajectory["language_instruction"] = format_instruction(
        trajectory['language_instruction'],
        "WindowX",
        "1",
        "Delta End-Effector",
        prompt_style="combined"
    )
    trajectory = relabel_actions(trajectory)
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("WindowX", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
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
    # trajectory["language_instruction"] = ""
    # iterate overa ll dicts in trajectory and print all keys
    trajectory["language_instruction"] = trajectory["observation"][
        "natural_language_instruction"
    ]
    # print(trajectory["observation"].keys())
    trajectory["language_instruction"] = format_instruction(
        trajectory["observation"]["natural_language_instruction"],
        "XARM",
        "1", 
        "Delta End-Effector",
        prompt_style="combined"
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"],
            trajectory["observation"]["gripper_closed"],
        ),
        axis=-1,
    )
    trajectory["robot_information"] = add_robot_information("WindowX", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(3, dtype=tf.int32)
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
    # trajectory["language_instruction"] = tf.fill(
    #    tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    #)  # delete uninformative language instruction
    # trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory["language_instruction"] = format_instruction(
        trajectory["observation"]["natural_language_instruction"],
        "XARM",
        "1",
        "Delta End-Effector",
        prompt_style="combined"
    )
    trajectory["robot_information"] = add_robot_information("Kuka iiwa", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(15, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Jaco 2", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Sawyer", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Hello Stretch", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(3, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(20, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("UR5", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(30, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS', 1, 'position')
    # trajectory['frequency'] = tf.constant(30, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("UR5", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Kuka iiwa", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :7]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(20, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(20, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(3, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


def cmu_franka_exploration_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :7]
    trajectory["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
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
    trajectory["robot_information"] = add_robot_information("xArm", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(2, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(20, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(20, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Google Robot", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("PR2", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("PR2", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory



def utokyo_xarm_bimanual_dataset_transform(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    trajectory["observation"]["proprio"] = trajectory["observation"][
        "end_effector_pose"
    ]
    trajectory["robot_information"] = add_robot_information("xArm", "delta end-effector", 2)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 2, 'position')
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
    trajectory["robot_information"] = add_robot_information("Multi-Robot", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(1, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("xArm", "absolute joint", 1)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS', 1, 'position')
    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS', 1, 'position')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
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
    trajectory["robot_information"] = add_robot_information("Sawyer", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32) # not true they dont know 
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
    trajectory["robot_information"] = add_robot_information("DLR SARA", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"][:, :6]
    trajectory["robot_information"] = add_robot_information("DLR SARA", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("DLR EDAN", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["ground_truth_states"]["EE"],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )
    trajectory["robot_information"] = add_robot_information("UR5", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(12.5, dtype=tf.int32)
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    # no proprio provided
    trajectory["observation"]["proprio"] = tf.zeros(
        (tf.shape(trajectory["action"])[0], 1), dtype=tf.float32
    )
    trajectory["robot_information"] = add_robot_information("Sawyer", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(20, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Kinova Gen3", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
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
    trajectory["robot_information"] = add_robot_information("PAMY2", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(20, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Fanuc Mate", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Hello Stretch", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
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
    print(trajectory.keys())
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("ViperX", "absolute joint", 2)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS_BIMANUAL', 2, 'position')

    # reprhase lang instruction 
    trajectory["language_instruction"] = format_instruction(
        trajectory["language_instruction"],
        robot_name="ViperX",
        action_space="joint position",
        number_arms="2",
        prompt_style='combined'
    )
    # trajectory['frequency'] = tf.constant(50, dtype=tf.int32)
    return trajectory


def mobile_aloha_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 50Hz to 10Hz
    factor = 5
    trajectory = tf.nest.map_structure(lambda x: x[::factor], trajectory)
    print(trajectory.keys())
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("ViperX", "absolute joint", 2)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS_BIMANUAL_NAV', 2, 'position')

    # reprhase lang instruction 
    trajectory["language_instruction"] = format_instruction(
        trajectory["language_instruction"],
        robot_name="ViperX",
        action_space="joint position",
        number_arms="2",
        prompt_style='combined'
    )
    # trajectory['frequency'] = tf.constant(50, dtype=tf.int32)
    return trajectory


def aloha_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relabel actions to convert from 50Hz to 10Hz
    factor = 5
    trajectory = tf.nest.map_structure(lambda x: x[::factor], trajectory)
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("ViperX", "absolute joint", 2)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS_BIMANUAL', 2, 'position')

    # reprhase lang instruction 
    trajectory["language_instruction"] = format_instruction(
        trajectory["global_instruction"],
        robot_name="ViperX",
        action_space="joint position",
        number_arms="2",
        prompt_style='combined'
    )
    # trajectory['frequency'] = tf.constant(50, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory['language_instruction'] = format_instruction(
        trajectory['language_instruction'],
        robot_name="Hello Stretch",
        action_space="delta end-effector",
        number_arms=1,
        prompt_style='combined'
    )
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    trajectory["robot_information"] = add_robot_information("Hello Stretch", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(4, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "absolute joint", 1)
    trajectory['action_space_index'] = get_action_space_index('JOINT_POS', 1, 'position')

    # reprhase lang instruction 
    trajectory["language_instruction"] = format_instruction(
        trajectory["language_instruction"],
        robot_name="Franka Panda",
        action_space="joint position",
        number_arms="1",
        prompt_style='combined'
    )

    # trajectory['frequency'] = tf.constant(5, dtype=tf.int32)
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    trajectory["action"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],  # Current TCP position in base frame
            tf.cast(trajectory["observation"]["gripper"][:, None], tf.float32),  # Current gripper state
        ),
        axis=-1,
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
    trajectory["robot_information"] = add_robot_information("Franka", "delta end-effector", 1)
    trajectory['action_space_index'] = get_action_space_index('EEF_POS', 1, 'velocity')
    # trajectory['frequency'] = tf.constant(10, dtype=tf.int32)
    return trajectory


OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge": bridge_transform,
    "bridge_dataset_without_single": bridge_transform,
    "kit_irl_real_kitchen_delta_des_joint_euler": kit_irl_dataset_abs_joint_transform, # old
    "kit_irl_real_kitchen_vis_delta_des_joint_euler": kit_irl_dataset_abs_joint_transform, # old
    "kit_irl_real_kitchen_vis": kit_irl_dataset_abs_joint_transform, # kit_irl_dataset_abs_joint_transform # kit_irl_dataset_joint_transform # kit_irl_dataset_abs_transform # kit_irl_dataset_transform
    "kit_irl_real_kitchen_lang": kit_irl_dataset_abs_joint_transform, # kit_irl_dataset_abs_joint_transform # kit_irl_dataset_joint_transform # kit_irl_dataset_abs_transform # kit_irl_dataset_transform
    "droid": droid_dataset_transform,
    "eef_droid": eef_droid_dataset_transform,
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
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
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
    "aloha_mobile": mobile_aloha_dataset_transform,
    "fmb_dataset": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "robo_set": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "aloha_play_dataset": aloha_play_dataset_transform,
}