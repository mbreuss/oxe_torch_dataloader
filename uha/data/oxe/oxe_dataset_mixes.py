"""Defines dataset mixtures and weights for the Open X-Embodiment Datasets."""


BRIDGE_MIX = [
    ("bridge_dataset", 1.0),
]

BRIDGE_MARCEL_MIX = [
    ("bridge", 1.0),
]

BRIDGE_PNP_MIX = [
    ("bridge_dataset_pnp_NILS", 1.0),
]
BRIDGE_PNP_MIX_NO_DRAWER_NO_V1 = [
    ("bridge_dataset_without_single", 1.0)
]

BRIDGE_PNP_MIX_NO_DRAWER_NO_V1_GEMINI = [
    ("bridge_dataset_pnp_no_drawer_machine_fold_GEMINI", 1.0)
]

BRIDGE_PNP_MIX_NO_DRAWER_V1 = [
    ("bridge_dataset_pnp_no_drawer_machine_fold_with_v1_NILS", 1.0)
]

BRIDGE_ALL_GT_MIX = [
    ("bridge_dataset_full_NILS", 1.0),
]

SIMPLER_ENV = [
  ("bridge_dataset", 1.0),
  ("fractal20220817_data", 0.6),
]

KIT_IRL_REAL_KITCHEN_MIX = [
  ("kit_irl_real_kitchen_lang", 1.0),
  ("kit_irl_real_kitchen_vis", 1.0),
]

REAL_KITCHEN_DROID_CO_TRAINING = [
  ("kit_irl_real_kitchen_lang", 1.0),
  ("droid", 0.00093374110191941499546243790359),
]

JOINT_MIX = [
  ("robo_set", 0.5),
  ("droid", 0.04),
  ("aloha_mobile", 1.0),
  ("berkeley_mvp_converted_externally_to_rlds", 1.0),
  ("berkeley_rpt_converted_externally_to_rlds", 1.0),
  ("toto", 1.0), # only joint proprio, no action, but we can use proprio as action
  ("kit_irl_real_kitchen_delta_des_joint_euler", 6.2),
  ("kit_irl_real_kitchen_vis_delta_des_joint_euler", 1.333),
]

FRAKTAL_MIX = [
    ("fractal20220817_data", 1.0),
]

RT_X_MIX = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    # ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 2.0),
    ("berkeley_cable_routing", 3.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 5.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
]


OXE_FRANKA_MIX = [
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
    # ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
]

SMALL_TEST_SET = [
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
]

OXE_MAGIC_SOUP_LANG_ONLY = [
    ("fractal20220817_data", 0.54087122203),
    # ("bridge_dataset", 1.0),
    ("taco_play", 2.0),
    ("jaco_play", 1.0),
    ("roboturk", 2.0),
    ("viola", 2.0),
    ("berkeley_autolab_ur5", 2.0),
    ("language_table", 0.1),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 2.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 0.1),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 2.0),
    ("bc_z", 0.2),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
]

OXE_MAGIC_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    # ("bridge_dataset", 1.0),
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
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
]


OXE_FLEX_ACT_SOUP = [
    ("fractal20220817_data", 0.54087122203),
    ("kuka", 0.8341046294),
    # ("bridge_dataset", 1.0),
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
    # ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    # ("uiuc_d3field", 1.0),  --> somehow raw data is broken
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 2.0),
    ("cmu_stretch", 1.0),
    ("gnm_dataset", 1.0),
    ("aloha_static_dataset", 3.0),
    # ("aloha_dagger_dataset", 1.0),
    ("aloha_mobile", 2.0),
    # ("fmb_dataset", 1.0),
    ("dobbe", 1.0),
    ("robo_set", 0.5),
    ("rh20t", 0.5),
]


OXE_FULL_MIX = [
    ("fractal20220817_data", 1.0),
    ("kuka", 1.0),
    # ("bridge_dataset", 1),
    ("taco_play", 1.0),
    ("jaco_play", 1.0),
    ("berkeley_cable_routing", 1.0),
    ("roboturk", 1.0),
    ("nyu_door_opening_surprising_effectiveness", 1.0),
    ("viola", 1.0),
    ("berkeley_autolab_ur5", 1.0),
    ("toto", 1.0),
    ("language_table", 1.0),
    ("columbia_cairlab_pusht_real", 1.0),
    ("stanford_kuka_multimodal_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_rot_dataset_converted_externally_to_rlds", 1.0),
    ("stanford_hydra_dataset_converted_externally_to_rlds", 1.0),
    ("austin_buds_dataset_converted_externally_to_rlds", 1.0),
    ("nyu_franka_play_dataset_converted_externally_to_rlds", 1.0),
    ("maniskill_dataset_converted_externally_to_rlds", 1.0),
    ("furniture_bench_dataset_converted_externally_to_rlds", 1.0),
    ("cmu_franka_exploration_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_kitchen_dataset_converted_externally_to_rlds", 1.0),
    ("ucsd_pick_and_place_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sailor_dataset_converted_externally_to_rlds", 1.0),
    ("austin_sirius_dataset_converted_externally_to_rlds", 1.0),
    ("bc_z", 1.0),
    ("utokyo_pr2_opening_fridge_converted_externally_to_rlds", 1.0),
    ("utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_pick_and_place_converted_externally_to_rlds", 1.0),
    ("utokyo_xarm_bimanual_converted_externally_to_rlds", 1.0),
    ("robo_net", 1.0),
    ("berkeley_mvp_converted_externally_to_rlds", 1.0),
    ("berkeley_rpt_converted_externally_to_rlds", 1.0),
    ("kaist_nonprehensile_converted_externally_to_rlds", 1.0),
    ("stanford_mask_vit_converted_externally_to_rlds", 1.0),
    ("tokyo_u_lsmo_converted_externally_to_rlds", 1.0),
    ("dlr_sara_pour_converted_externally_to_rlds", 1.0),
    ("dlr_sara_grid_clamp_converted_externally_to_rlds", 1.0),
    ("dlr_edan_shared_control_converted_externally_to_rlds", 1.0),
    ("asu_table_top_converted_externally_to_rlds", 1.0),
    ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ("imperialcollege_sawyer_wrist_cam", 1.0),
    ("iamlab_cmu_pickup_insert_converted_externally_to_rlds", 1.0),
    ("uiuc_d3field", 1.0),
    ("utaustin_mutex", 1.0),
    ("berkeley_fanuc_manipulation", 1.0),
    ("cmu_playing_with_food", 1.0),
    ("cmu_play_fusion", 1.0),
    ("cmu_stretch", 1.0),
    ("gnm_dataset", 1.0),
]

KIT_IRL_REAL_KITCHEN_DELTA_DES_JOINT_EULER = [
    ("kit_irl_real_kitchen_delta_des_joint_euler", 1.0),
]
KIT_IRL_REAL_KITCHEN_VIS_DELTA_DES_JOINT_EULER = [
    ("kit_irl_real_kitchen_vis_delta_des_joint_euler", 1.0),
]
KIT_IRL_REAL_KITCHEN_LANG = [
    ("kit_irl_real_kitchen_lang", 1.0),
]
KIT_IRL_REAL_KITCHEN_VIS = [
    ("kit_irl_real_kitchen_vis", 1.0),
]

# === LIBERO Datasets (Modified Versions) ===
LIBERO_SPATIAL_NO_NOOPS = [
  ("libero_spatial_no_noops", 1.0),
]
LIBERO_OBJECT_NO_NOOPS = [
  ("libero_object_no_noops", 1.0),
]
LIBERO_GOAL_NO_NOOPS = [
  ("libero_goal_no_noops", 1.0),
]
LIBERO_10_NO_NOOPS = [
  ("libero_10_no_noops", 1.0),
]
LIBERO_ALL = [
  ("libero_spatial_no_noops", 1.0),
  ("libero_object_no_noops", 1.0),
  ("libero_goal_no_noops", 1.0),
  ("libero_10_no_noops", 1.0),
]
SIMPLER_LIBERO_ALL = [
  ("libero_spatial_no_noops", 8.0),
  ("libero_object_no_noops", 8.0),
  ("libero_goal_no_noops", 8.0),
  ("libero_10_no_noops", 8.0),
  ("bridge_dataset", 1.0),
  ("fractal20220817_data", 1.0),
]

MO_ALL = [
  ("libero_spatial_no_noops", 10.0),
  ("libero_object_no_noops", 10.0),
  ("libero_goal_no_noops", 10.0),
  ("libero_10_no_noops", 10.0),
  ("bridge_dataset", 1.0),
  ("fractal20220817_data", 0.6),
  ("robo_set", 0.5),
  ("kit_irl_real_kitchen_lang", 10.0),
  ("aloha_mobile", 0.5),
  # ("aloha_dagger_dataset", 3.0),
  ("bc_z", 0.1),
  ("berkeley_rpt_converted_externally_to_rlds", 1.0),
  ("berkeley_mvp_converted_externally_to_rlds", 0.5),
  ("dobbe", 0.5),
  # ("droid", 0.04),
]

MO_ALL = [
  # ("cmu_play_fusion", 1.0),
  # ("berkeley_mvp_converted_externally_to_rlds", 1.0),
  # ("bc_z", 0.2),
  # ("kit_irl_real_kitchen_lang", 10.0),
  ("dobbe", 1.0),
  # ("aloha_mobile", 2.0),
 ]

OXE_NAMED_MIXES = {
    "fraktal": FRAKTAL_MIX,
    "simpler_env": SIMPLER_ENV,
    "real_kitchen_mix": KIT_IRL_REAL_KITCHEN_MIX,
    "real_kitchen_lang": KIT_IRL_REAL_KITCHEN_LANG,
    "real_kitchen_vis": KIT_IRL_REAL_KITCHEN_VIS,
    "finetune_data_delta_des_joint_euler": KIT_IRL_REAL_KITCHEN_DELTA_DES_JOINT_EULER,
    "finetune_data_vis_delta_des_joint_euler": KIT_IRL_REAL_KITCHEN_VIS_DELTA_DES_JOINT_EULER,
    "real_kitchen_droid_co_training": REAL_KITCHEN_DROID_CO_TRAINING,
    "joint_mix": JOINT_MIX,
    "bridge_marcel": BRIDGE_MARCEL_MIX,
    "bridge_pnp": BRIDGE_PNP_MIX,
    "bridge_pnp_no_drawer_no_v1": BRIDGE_PNP_MIX_NO_DRAWER_NO_V1,
    "bridge_pnp_no_drawer_v1": BRIDGE_PNP_MIX_NO_DRAWER_V1,
    "bridge_pnp_no_drawer_no_v1_gemini" : BRIDGE_PNP_MIX_NO_DRAWER_NO_V1_GEMINI,
    "bridge_all_gt": BRIDGE_ALL_GT_MIX,
    "bridge": BRIDGE_MIX,
    "libero_spatial": LIBERO_SPATIAL_NO_NOOPS,
    "libero_object": LIBERO_OBJECT_NO_NOOPS,
    "libero_goal": LIBERO_GOAL_NO_NOOPS,
    "libero_10": LIBERO_10_NO_NOOPS,
    "rtx": RT_X_MIX,
    "rtx_franka": RT_X_MIX + OXE_FRANKA_MIX,
    "oxe_magic_soup": OXE_MAGIC_SOUP,
    "oxe_flex_act_soup": OXE_FLEX_ACT_SOUP,
    "oxe_magic_soup_lang_only": OXE_MAGIC_SOUP_LANG_ONLY,
    "small_test": SMALL_TEST_SET,
    "libero_all": LIBERO_ALL,
    "simulation_all": SIMPLER_LIBERO_ALL,
    "mo_all": MO_ALL,
}