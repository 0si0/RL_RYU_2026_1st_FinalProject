# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path
from gr.asset.shadow_hand import SHADOW_HAND_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[6]

# Main sequence
SEQ_PATH = str(PROJECT_ROOT / "data" / "HOCAP" / "sequence1" / "sequence1.pt")
OBJ_PATH = str(PROJECT_ROOT / "data" / "HOCAP" / "object" / "sequence1.usd")
END_FRAME = 250

# # Optional sequence
# SEQ_PATH = str(PROJECT_ROOT / "data" / "HOCAP" / "sequence2" / "sequence2.pt")
# OBJ_PATH = str(PROJECT_ROOT / "data" / "HOCAP" / "object" / "sequence2.usd")
# END_FRAME = 660

# # Optional sequence
# SEQ_PATH = str(PROJECT_ROOT / "data" / "HOCAP" / "sequence3" / "sequence3.pt")
# OBJ_PATH = str(PROJECT_ROOT / "data" / "HOCAP" / "object" / "sequence3.usd")
# END_FRAME = 510


@configclass
class GrEnvCfg(DirectRLEnvCfg):
    play = False
    asymmetric_obs = False
    
    # TODO: Match this dimension to the observation vector built in gr_env.py.
    observation_space = 302

    # env
    decimation = 4
    obs_type = "full"

    table_upper_z = 0.4
    table_pos_z = -0.1

    hand_mount = 'robot0_hand_mount'
    root_body = 'robot0_palm'


    body_to_kpts_except_fingertips = [0, 5, 20, 22, 1, 11, 16, 2, 12, 17, 3, 13, 18, 9, 19, 21]
    MANO_kpts = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    MANO_kpts_except_fingertips = [0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    MANO_fingertips = [4, 8, 12, 16, 20]
    MANO_finger_joint_groups = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    MANO_topology_joints = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    MANO_rigids = [0, 5, 9, 13]
    MANO_anchors = [0, 5, 9, 13, 17]
    lookahead_reference_steps = (1, 5, 10)

    seq_ref_path = SEQ_PATH
    obj_path = OBJ_PATH
    start_frame = 0
    end_frame = END_FRAME
    
    action_fps = 30
    episode_length = max(1, end_frame - start_frame)
    num_frame_chunk = episode_length
    episode_length_s = (((num_frame_chunk)*10)//action_fps)/10.0
    warm_up_epochs = 0


    # PD controller gains
    K_pos = 4000
    K_rot = 160

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / (action_fps * decimation),
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_patch_count=5 * 2**17,
            gpu_total_aggregate_pairs_capacity=2 ** 22,
        ),
    )

    # robot
    robot_cfg: ArticulationCfg = SHADOW_HAND_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    
    # camera
    viewer: ViewerCfg = ViewerCfg(
        eye=(3.0, 3.0, 2.0),
        lookat=(1.0, 1.0, 0.2),
    )

    num_revolving_joints = 22
    actuated_joint_names = [
        "robot0_FFJ3",
        "robot0_FFJ2",
        "robot0_FFJ1",
        "robot0_MFJ3",
        "robot0_MFJ2",
        "robot0_MFJ1",
        "robot0_RFJ3",
        "robot0_RFJ2",
        "robot0_RFJ1",
        "robot0_LFJ4",
        "robot0_LFJ3",
        "robot0_LFJ2",
        "robot0_LFJ1",
        "robot0_THJ4",
        "robot0_THJ3",
        "robot0_THJ2",
        "robot0_THJ1",
        "robot0_THJ0",
    ]
    fingertip_body_names = [
        "robot0_thdistal",
        "robot0_ffdistal",
        "robot0_mfdistal",
        "robot0_rfdistal",
        "robot0_lfdistal",
    ]

    end_joint_names = [
        "robot0_FFJ3",
        "robot0_MFJ3",
        "robot0_RFJ3",
        "robot0_LFJ3",
        "robot0_THJ3",
    ]
    
    num_dof = len(actuated_joint_names)
    action_space = 9 + num_dof # trans + rotation + joint
    state_space = 0
    
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=obj_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=2.5, replicate_physics=True, clone_in_fabric=False
    )
    
    goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_markers",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/sphere.usd",
                scale=(0.03, 0.03, 0.03),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0),
                ),
            )
        }
    )

    debug_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/debug_markers",
        markers={
            "debug": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/sphere.usd",
                scale=(0.03, 0.03, 0.03),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0),
                ),
            )
        }
    )

    table_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.5, 1.5, 2*(table_upper_z-table_pos_z)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, table_pos_z)),
    )

    action_dt = 1 / action_fps
    
    action_penalty_scale = -0.002
    dof_penalty_scale = -0.001

    hand_reward_weight = 0.7
    hand_pos_reward_weight = 0.6
    hand_anchor_reward_weight = 0.6
    hand_obj_offset_reward_weight = 0.6
    anchor_obj_offset_reward_weight = 0.8
    hand_dof_reward_weight = 0.25
    hand_rot_reward_weight = 0.2
    finger_shape_reward_weight = 0.15
    finger_topology_reward_weight = 0.35
    fingertip_reward_weight = 0.4
    fingertip_obj_proximity_reward_weight = 0.75
    fingertip_obj_offset_reward_weight = 0.8
    contact_reward_weight = 1.5
    obj_pos_reward_weight = 2.0
    obj_rot_reward_weight = 1.0
    obj_vel_reward_weight = 0.2

    hand_reward_scale = 25.0
    hand_pos_reward_scale = 20.0
    hand_anchor_reward_scale = 15.0
    hand_obj_offset_reward_scale = 12.0
    anchor_obj_offset_reward_scale = 15.0
    hand_dof_reward_scale = 4.0
    hand_rot_reward_scale = 2.0
    finger_shape_reward_scale = 25.0
    finger_topology_reward_scale = 25.0
    finger_spread_reward_scale = 35.0
    finger_spread_collapse_margin = 0.0075
    finger_spread_min_ref_distance = 0.015
    finger_direction_error_weight = 0.02
    finger_shape_contact_decay = 0.05
    finger_topology_contact_decay = 0.30
    anchor_rotation_gate_scale = 20.0
    fingertip_reward_scale = 40.0
    fingertip_obj_proximity_reward_scale = 8.0
    fingertip_obj_offset_reward_scale = 15.0
    proximity_gate_scale = 4.0
    object_reward_gate_base = 0.05
    contact_force_reward_weight = 0.2
    contact_count_reward_weight = 0.7
    contact_sustain_reward_weight = 0.6
    stable_grasp_reward_weight = 0.8
    transport_support_reward_weight = 1.0
    obj_future_dir_reward_weight = 0.15
    grasped_hand_ref_reward_weight = 0.6
    early_imitation_reward_bonus = 0.6
    early_episode_tracking_frames = 70.0
    early_episode_tracking_bonus = 0.8
    early_lag_penalty_weight = 1.0
    approach_imitation_bonus = 0.5
    grasp_object_bonus = 0.5
    manipulation_task_bonus = 0.6
    manipulation_imitation_bonus = 0.4
    successful_grasp_dof_bonus_weight = 0.25
    successful_grasp_spread_bonus_mix = 0.15
    pre_contact_pose_bonus_weight = 0.35
    no_contact_mano_imitation_floor = 0.25
    object_relative_reward_base = 0.35
    mid_object_relative_reward_bonus = 1.0
    late_task_reward_bonus = 1.0
    reward_curriculum_steps = 18000
    anchor_object_gate_floor = 0.35
    no_contact_grace_steps = 60.0
    no_contact_late_reward_floor = 0.05
    no_grasp_rotation_penalty_weight = 0.10
    contact_force_threshold = 0.005
    target_contact_fingers = 3.0
    contact_reward_max_force = 0.2
    force_dominance_limit = 0.65
    reference_contact_proximity_scale = 20.0
    obj_future_dir_min_distance = 0.02
    contact_topk_fingers = 3
    contact_sustain_target_steps = 8.0
    obj_pos_reward_scale = 20.0
    obj_delta_reward_scale = 20.0
    obj_rot_reward_scale = 2.0
    obj_vel_reward_scale = 2.0
    obj_angvel_reward_scale = 0.2

    hand_terminate_threshold = 0.35
    obj_terminate_threshold = 0.35
    obj_rot_terminate_threshold = 1.0
    obj_rot_terminate_after_steps = 80.0
    no_grasp_terminate_after_steps = 110.0
    no_grasp_terminate_grace_steps = 70.0
    random_reference_phase_sampling = True
    reference_phase_min_remaining_steps = 45
    reference_phase_frame0_ratio = 0.60
    reference_phase_uniform_ratio = 0.30
    success_biased_phase_sampling = True
    success_phase_weight_decay = 0.995
    success_phase_weight_gain = 0.2
    success_phase_spread = 4

    act_moving_average = 0.5
    global_moving_average = 0.4

    adaptive_uniform_ratio = 0.1
    adaptive_alpha = 0.001
    vel_obs_scale = 0.2

    def __post_init__(self):
        super().__post_init__()
        for finger in self.fingertip_body_names:
            setattr(
                self.scene,
                f'contact_sensor_{finger}',
                ContactSensorCfg(
                    prim_path=f'{{ENV_REGEX_NS}}/Robot/{finger}',
                    update_period=0.0,
                    history_length=6,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
