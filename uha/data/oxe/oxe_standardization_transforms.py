
"""Dataset standardization transforms registry with complete dataset coverage."""

from typing import Dict, Any
import logging

import tensorflow as tf

from .transforms.factory import TransformFactory
from .transforms.base import BaseTransform

logger = logging.getLogger(__name__)



class StandardizationRegistry:
    """Registry for dataset standardization transforms."""
    
    def __init__(self):
        self._transforms: Dict[str, BaseTransform] = {}
        self._initialize_transforms()
    
    def _initialize_transforms(self):
        """Initialize standard transforms for all datasets."""
        try:
            self._transforms.update({
                # Bridge datasets
                "bridge": TransformFactory.create(
                    "franka",
                    robot_name="WindowX",
                    action_space="delta end-effector"
                ),
                "bridge_dataset_without_single": TransformFactory.create(
                    "franka",
                    robot_name="WindowX",
                    action_space="delta end-effector"
                ),

                # Kitchen datasets
                "kit_irl_real_kitchen_delta_des_joint_euler": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="absolute joint"
                ),
                "kit_irl_real_kitchen_vis_delta_des_joint_euler": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="absolute joint"
                ),
                "kit_irl_real_kitchen_vis": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="absolute joint"
                ),
                "kit_irl_real_kitchen_lang": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="absolute joint"
                ),

                # Droid dataset
                "droid": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="absolute joint",
                    gripper_threshold=0.95
                ),

                # Bridge dataset variants
                "bridge_dataset": TransformFactory.create(
                    "franka",
                    robot_name="WindowX",
                    action_space="delta end-effector"
                ),
                
                # RT1 and related datasets
                "fractal20220817_data": TransformFactory.create(
                    "franka",
                    robot_name="RT1",
                    action_space="delta end-effector"
                ),
                "kuka": TransformFactory.create(
                    "franka",
                    robot_name="Kuka iiwa",
                    action_space="delta end-effector"
                ),

                # TACO dataset
                "taco_play": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),

                # Jaco dataset
                "jaco_play": TransformFactory.create(
                    "franka",
                    robot_name="Jaco 2",
                    action_space="delta end-effector"
                ),

                # Berkeley datasets
                "berkeley_cable_routing": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "berkeley_autolab_ur5": TransformFactory.create(
                    "ur5",
                    robot_name="UR5",
                    action_space="delta end-effector"
                ),
                "berkeley_mvp_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="xArm",
                    action_space="absolute joint"
                ),
                "berkeley_rpt_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="absolute joint"
                ),
                "berkeley_fanuc_manipulation": TransformFactory.create(
                    "franka",
                    robot_name="Fanuc Mate",
                    action_space="delta end-effector"
                ),

                # RoboTurk dataset
                "roboturk": TransformFactory.create(
                    "franka",
                    robot_name="Sawyer",
                    action_space="delta end-effector"
                ),

                # NYU datasets
                "nyu_door_opening_surprising_effectiveness": TransformFactory.create(
                    "franka",
                    robot_name="Hello Stretch",
                    action_space="delta end-effector"
                ),
                "nyu_franka_play_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "nyu_rot_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="xArm",
                    action_space="delta end-effector"
                ),

                # Viola dataset
                "viola": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),

                # Toto dataset
                "toto": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="absolute joint"
                ),

                # Language Table dataset
                "language_table": TransformFactory.create(
                    "franka",
                    robot_name="xArm",
                    action_space="delta end-effector"
                ),

                # Stanford datasets
                "stanford_hydra_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Kuka iiwa",
                    action_space="delta end-effector"
                ),
                "stanford_robocook_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),

                # UCSD datasets
                "ucsd_kitchen_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="xArm",
                    action_space="delta end-effector"
                ),
                "ucsd_pick_and_place_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="xArm",
                    action_space="delta end-effector"
                ),

                # Austin datasets
                "austin_buds_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "austin_sailor_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "austin_sirius_dataset_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),

                # BCZ dataset
                "bc_z": TransformFactory.create(
                    "franka",
                    robot_name="Google Robot",
                    action_space="delta end-effector"
                ),

                # Tokyo datasets
                "tokyo_u_lsmo_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Custom Robot",
                    action_space="delta end-effector"
                ),
                "utokyo_pr2_opening_fridge_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="PR2",
                    action_space="delta end-effector"
                ),
                "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="PR2",
                    action_space="delta end-effector"
                ),
                "utokyo_xarm_pick_and_place_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="xArm",
                    action_space="delta end-effector"
                ),
                "utokyo_xarm_bimanual_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="xArm",
                    action_space="delta end-effector",
                    num_arms=2
                ),

                # RoboNet dataset
                "robo_net": TransformFactory.create(
                    "franka",
                    robot_name="Multi-Robot",
                    action_space="delta end-effector"
                ),

                # Kaist dataset
                "kaist_nonprehensile_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),

                # Stanford Mask ViT dataset
                "stanford_mask_vit_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Sawyer",
                    action_space="delta end-effector"
                ),

                # DLR datasets
                "dlr_sara_pour_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="DLR SARA",
                    action_space="delta end-effector"
                ),
                "dlr_sara_grid_clamp_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="DLR SARA",
                    action_space="delta end-effector"
                ),
                "dlr_edan_shared_control_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="DLR EDAN",
                    action_space="delta end-effector"
                ),

                # ASU dataset
                "asu_table_top_converted_externally_to_rlds": TransformFactory.create(
                    "ur5",
                    robot_name="UR5",
                    action_space="delta end-effector"
                ),

                # Imperial College dataset
                "imperialcollege_sawyer_wrist_cam": TransformFactory.create(
                    "franka",
                    robot_name="Sawyer",
                    action_space="delta end-effector"
                ),

                # IAMLAB CMU dataset
                "iamlab_cmu_pickup_insert_converted_externally_to_rlds": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),

                # UIUC dataset
                "uiuc_d3field": TransformFactory.create(
                    "franka",
                    robot_name="Kinova Gen3",
                    action_space="delta end-effector"
                ),

                # UT Austin dataset
                "utaustin_mutex": TransformFactory.create(
                    "franka",
                    robot_name="PAMY2",
                    action_space="delta end-effector"
                ),

                # CMU datasets
                "cmu_playing_with_food": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "cmu_play_fusion": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "cmu_stretch": TransformFactory.create(
                    "franka",
                    robot_name="Hello Stretch",
                    action_space="delta end-effector"
                ),

                # GNM dataset
                "gnm_dataset": TransformFactory.create(
                    "franka",
                    robot_name="Navigation Robot",
                    action_space="2D waypoint"
                ),

                # ALOHA datasets
                "aloha_static_dataset": TransformFactory.create(
                    "bimanual",
                    robot_name="ViperX",
                    action_space="absolute joint",
                    num_arms=2
                ),
                "aloha_dagger_dataset": TransformFactory.create(
                    "bimanual",
                    robot_name="ViperX",
                    action_space="absolute joint",
                    num_arms=2
                ),
                "aloha_mobile": TransformFactory.create(
                    "bimanual",
                    robot_name="ViperX",
                    action_space="absolute joint",
                    num_arms=2
                ),

                # FMB dataset
                "fmb_dataset": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),

                # Dobbe dataset
                "dobbe": TransformFactory.create(
                    "franka",
                    robot_name="Hello Stretch",
                    action_space="delta end-effector"
                ),

                # RoboSet dataset
                "robo_set": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="absolute joint"
                ),

                # RH20T dataset
                "rh20t": TransformFactory.create(
                    "franka",
                    robot_name="RH20T",
                    action_space="delta end-effector"
                ),

                # Mujoco dataset
                "mujoco_manip": TransformFactory.create(
                    "franka",
                    robot_name="Mujoco Robot",
                    action_space="delta end-effector"
                ),

                # LIBERO datasets
                "libero_spatial_no_noops": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "libero_object_no_noops": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "libero_goal_no_noops": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
                "libero_10_no_noops": TransformFactory.create(
                    "franka",
                    robot_name="Franka",
                    action_space="delta end-effector"
                ),
            })
            
        except Exception as e:
            logger.error(f"Failed to initialize transforms: {str(e)}")
            raise

    def get_transform(self, dataset_name: str) -> BaseTransform:
        """Get transform for dataset."""
        if dataset_name not in self._transforms:
            raise ValueError(f"No transform registered for dataset: {dataset_name}")
        return self._transforms[dataset_name]

    def register_transform(self, dataset_name: str, transform: BaseTransform):
        """Register new transform."""
        self._transforms[dataset_name] = transform


# Create global registry instance
STANDARDIZATION_REGISTRY = StandardizationRegistry()


def get_transform_fn(dataset_name: str):
    """Get transform function for dataset."""
    try:
        transform = STANDARDIZATION_REGISTRY.get_transform(dataset_name)
        return transform
    except Exception as e:
        logger.error(f"Error getting transform for {dataset_name}: {str(e)}")
        raise


# Export transform mapping
OXE_STANDARDIZATION_TRANSFORMS = {
    name: transform 
    for name, transform in STANDARDIZATION_REGISTRY._transforms.items()
}
