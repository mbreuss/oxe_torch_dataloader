"""Dataset mixture definitions and management for Open X-Embodiment datasets."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetMix:
    """Configuration for a dataset mixture."""
    name: str
    datasets: List[Tuple[str, float]]
    description: Optional[str] = None

    def __post_init__(self):
        """Validate mixture configuration."""
        if not self.datasets:
            raise ValueError(f"Dataset mix '{self.name}' must contain at least one dataset")
        if any(weight <= 0 for _, weight in self.datasets):
            raise ValueError(f"All weights in mix '{self.name}' must be positive")


class DatasetMixRegistry:
    """Registry for dataset mixtures."""

    def __init__(self):
        self._mixes: Dict[str, DatasetMix] = {}
        self._initialize_mixes()

    def _initialize_mixes(self):
        """Initialize all dataset mixtures."""
        self._mixes.update({
            # Bridge Mixes
            "bridge": DatasetMix(
                name="bridge",
                datasets=[("bridge_dataset", 1.0)],
                description="Standard Bridge dataset"
            ),

            "bridge_marcel": DatasetMix(
                name="bridge_marcel",
                datasets=[("bridge", 1.0)],
                description="Marcel's Bridge dataset version"
            ),

            # Environment Mixes
            "simpler_env": DatasetMix(
                name="simpler_env",
                datasets=[
                    ("bridge_dataset", 1.0),
                    ("fractal20220817_data", 0.6)
                ],
                description="Simplified environment mix"
            ),

            # Kitchen Mixes
            "kit_irl_real_kitchen_mix": DatasetMix(
                name="kit_irl_real_kitchen_mix",
                datasets=[
                    ("kit_irl_real_kitchen_lang", 1.0),
                    ("kit_irl_real_kitchen_vis", 1.0)
                ],
                description="Kitchen dataset mix with language and visual data"
            ),

            # Co-training Mixes
            "real_kitchen_droid_co_training": DatasetMix(
                name="real_kitchen_droid_co_training",
                datasets=[
                    ("kit_irl_real_kitchen_lang", 1.0),
                    ("droid", 0.00093374110191941499546243790359)
                ],
                description="Kitchen and Droid co-training mix"
            ),

            # Joint Control Mixes
            "joint_mix": DatasetMix(
                name="joint_mix",
                datasets=[
                    ("robo_set", 0.5),
                    ("droid", 0.04),
                    ("aloha_mobile", 1.0),
                    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
                    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
                    ("toto", 1.0),
                    ("kit_irl_real_kitchen_delta_des_joint_euler", 6.2),
                    ("kit_irl_real_kitchen_vis_delta_des_joint_euler", 1.333)
                ],
                description="Mix of datasets with joint control"
            ),

            # Franka Robot Mixes
            "oxe_franka_mix": DatasetMix(
                name="oxe_franka_mix",
                datasets=[
                    ("taco_play", 1.0),
                    ("berkeley_cable_routing", 1.0),
                    ("viola", 1.0),
                    ("toto", 1.0),
                    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
                    ("austin_buds_dataset_converted_externally_to_rlds", 3.0),
                    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
                    ("maniskill_dataset_converted_externally_to_rlds", 0.1),
                    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
                    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 5.0),
                    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
                    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
                    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
                    ("kaist_nonprehensile_converted_externally_to_rlds", 3.0),
                    ("stanford_robocook_converted_externally_to_rlds", 1.0),
                    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
                    ("utaustin_mutex", 1.0),
                    ("cmu_play_fusion", 1.0)
                ],
                description="Mix focused on Franka robot datasets"
            ),
            # Mixed Objects (MO) Mix
            "MO": DatasetMix(
                name="MO",
                datasets=[
                    ("libero_spatial_no_noops", 0.051870),
                    ("libero_object_no_noops", 0.065593),
                    ("libero_goal_no_noops", 0.050961),
                    ("libero_10_no_noops", 0.099362),
                    ("bridge_dataset", 0.268742),
                    ("fractal20220817_data", 0.463472)
                ],
                description="Mixed Objects dataset combining LIBERO, Bridge and Fractal data"
            ),
            # Comprehensive Mixes
            "oxe_flex_act_soup": DatasetMix(
                name="oxe_flex_act_soup",
                datasets=[
                    ("fractal20220817_data", 0.54087122203),
                    ("kuka", 0.8341046294),
                    ("taco_play", 2.0),
                    ("jaco_play", 1.0),
                    ("berkeley_cable_routing", 1.0),
                    ("roboturk", 2.0),
                    ("nyu_door_opening_surprising_effectiveness", 1.0),
                    ("viola", 2.0),
                    ("berkeley_autolab_ur5", 2.0),
                    ("toto", 1.0),
                    ("language_table", 0.1),
                    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
                    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
                    ("nyu_franka_play_dataset_converted_externally_to_rlds", 3.0),
                    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
                    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
                    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
                    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
                    ("bc_z", 0.2),
                    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
                    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
                    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
                    ("utaustin_mutex", 1.0),
                    ("berkeley_fanuc_manipulation", 2.0),
                    ("cmu_stretch", 1.0),
                    ("gnm_dataset", 1.0),
                    ("aloha_static_dataset", 3.0),
                    ("aloha_mobile", 2.0),
                    ("dobbe", 1.0),
                    ("robo_set", 0.5),
                    ("rh20t", 0.5)
                ],
                description="Comprehensive mix with flexible action spaces"
            ),

            # LIBERO Mixes
            "libero_all": DatasetMix(
                name="libero_all",
                datasets=[
                    ("libero_spatial_no_noops", 1.0),
                    ("libero_object_no_noops", 1.0),
                    ("libero_goal_no_noops", 1.0),
                    ("libero_10_no_noops", 1.0)
                ],
                description="All LIBERO datasets"
            ),

            "simulation_all": DatasetMix(
                name="simulation_all",
                datasets=[
                    ("libero_spatial_no_noops", 8.0),
                    ("libero_object_no_noops", 8.0),
                    ("libero_goal_no_noops", 8.0),
                    ("libero_10_no_noops", 8.0),
                    ("bridge_dataset", 1.0),
                    ("fractal20220817_data", 1.0)
                ],
                description="All simulation datasets with LIBERO emphasis"
            ),

            # Testing Mixes
            "small_test": DatasetMix(
                name="small_test",
                datasets=[
                    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
                    ("austin_buds_dataset_converted_externally_to_rlds", 1.0)
                ],
                description="Small test mix for development"
            )
        })

    def get_mix(self, mix_name: str) -> List[Tuple[str, float]]:
        """Get dataset mixture by name."""
        if mix_name not in self._mixes:
            raise ValueError(f"Unknown dataset mix: {mix_name}")
        return self._mixes[mix_name].datasets

    def register_mix(self, mix: DatasetMix):
        """Register a new dataset mixture."""
        if mix.name in self._mixes:
            logger.warning(f"Overwriting existing mix: {mix.name}")
        self._mixes[mix.name] = mix

    def list_mixes(self) -> List[str]:
        """List all available mixture names."""
        return sorted(self._mixes.keys())

    def get_mix_info(self, mix_name: str) -> DatasetMix:
        """Get full mixture information including description."""
        if mix_name not in self._mixes:
            raise ValueError(f"Unknown dataset mix: {mix_name}")
        return self._mixes[mix_name]


# Create global registry instance
MIX_REGISTRY = DatasetMixRegistry()

# Export mix configurations
OXE_NAMED_MIXES = {
    name: mix.datasets 
    for name, mix in MIX_REGISTRY._mixes.items()
}