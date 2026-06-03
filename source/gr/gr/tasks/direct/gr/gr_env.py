# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_angle_axis, quat_mul, quat_apply, saturate, matrix_from_quat, quat_from_matrix, euler_xyz_from_quat, quat_from_euler_xyz
from .gr_env_cfg import GrEnvCfg
from pxr import Usd, UsdPhysics
import omni.usd


class GrEnv(DirectRLEnv):
    cfg: GrEnvCfg

    def __init__(self, cfg: GrEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.inputs = torch.load(cfg.seq_ref_path, map_location="cpu")

        self.num_hand_dof = self.hand.num_joints

        self.num_kpts = len(self.cfg.MANO_kpts)
        self.finger_joint_groups = torch.tensor(self.cfg.MANO_finger_joint_groups, dtype=torch.long, device=self.device)
        self.topology_joint_indices = torch.tensor(self.cfg.MANO_topology_joints, dtype=torch.long, device=self.device)
        self.termination = not self.cfg.play
        self.play = self.cfg.play
        self.time_out = torch.zeros((self.num_envs, ), device=self.device).bool()
        self.episode_length = self.cfg.episode_length

        # list of joints, hand_bodies, fingertip_bodies, root, rigid bodies
        self.actuated_dof_indices = list()
        self.root_body = list()
        self.hand_bodies = list()
        self.hand_body_names = list()
        self.fingertip_bodies = list()

        for joint_name in self.cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        for i in range(len(self.hand.data.body_names)):
            if self.hand.data.body_names[i] != 'robot0_hand_mount':
                self.hand_body_names.append(self.hand.data.body_names[i])
                self.hand_bodies.append(i)
                if self.hand.data.body_names[i] == 'robot0_palm':
                    self.root_body.append(i)
        for body_name in self.cfg.fingertip_body_names:
            self.fingertip_bodies.append(self.hand_body_names.index(body_name))
        
        # num of joints, hand_bodies, fingertip_bodies, rigid bodies
        self.num_actuated_dof = len(self.actuated_dof_indices)
        self.num_hand_bodies = len(self.hand_bodies)
        self.num_fingertips = len(self.fingertip_bodies)
        
        # ref parameters
        self.hand_pos_ref = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot_ref = torch.zeros((self.num_envs, 4), device=self.device)
        self.hand_dof_ref = torch.zeros((self.num_envs, self.num_hand_dof), device=self.device)
        self.obj_pos_ref = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_rot_ref = torch.zeros((self.num_envs, 4), device=self.device)
        self.hand_rot_ref[:,0] = 1.0
        self.obj_rot_ref[:,0] = 1.0
        self.lookahead_ref_obs = torch.zeros(
            (self.num_envs, 33 * len(self.cfg.lookahead_reference_steps)),
            device=self.device,
        )
        self.object_lookahead_ref_obs = torch.zeros(
            (self.num_envs, 9 * len(self.cfg.lookahead_reference_steps)),
            device=self.device,
        )

        # object parameters
        self.obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.obj_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos_reset = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_rot_reset = torch.zeros((self.num_envs, 4), device=self.device)
        self.obj_rot_reset[:,0] = 1.0

        # hand parameters
        self.hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.hand_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_pos_reset = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot_reset = torch.zeros((self.num_envs, 4), device=self.device)
        self.hand_rot_reset[:,0] = 1.0
        self.hand_dof_pos_reset = torch.zeros((self.num_envs, self.num_hand_dof), device=self.device)
        self.hand_dof_pos = torch.zeros((self.num_envs, self.num_hand_dof), device=self.device)
        self.hand_dof_vel = torch.zeros((self.num_envs, self.num_hand_dof), device=self.device)

        self.hand_bodies_pos = torch.zeros((self.num_envs,self.num_hand_bodies,3), device=self.device)
        self.hand_bodies_rot = torch.zeros((self.num_envs,self.num_hand_bodies,4), device=self.device)
        self.hand_bodies_linvel = torch.zeros((self.num_envs,self.num_hand_bodies,3), device=self.device)
        self.hand_bodies_angvel = torch.zeros((self.num_envs,self.num_hand_bodies,3), device=self.device)

        self.hand_kpts_pos = torch.zeros((self.num_envs, self.num_kpts, 3), device=self.device)

        # fingertip parameters
        self.fingertip_pos = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)
        self.fingertip_normal = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)
        self.fingertip_normal[:, 1:, 1] = -1
        self.fingertip_normal[:, 0, 0] = -1
        self.fingertip_rot = torch.zeros((self.num_envs,self.num_fingertips,4), device=self.device)
        self.fingertip_linvel = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)
        self.fingertip_angvel = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)

        # body to keypoints
        self.fingertip_offset = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)
        self.fingertip_offset[:, 0, :] = torch.tensor([-0.0085, 0.0, 0.02], device=self.device)
        self.fingertip_offset[:, 1, :] = torch.tensor([0.0, -0.006, 0.0175], device=self.device)
        self.fingertip_offset[:, 2, :] = torch.tensor([0.0, -0.006, 0.0175], device=self.device)
        self.fingertip_offset[:, 3, :] = torch.tensor([0.0, -0.006, 0.0175], device=self.device)
        self.fingertip_offset[:, 4, :] = torch.tensor([0.0, -0.006, 0.0175], device=self.device)

        # fingertip force
        self.fingertip_contact_forces = torch.zeros((self.num_envs, self.num_fingertips,3), device=self.device)
        self.fingertip_contact_forces_buf = torch.zeros((self.num_envs, 3, self.num_fingertips), device=self.device)
        self.fingertip_contact_force_norm = torch.zeros((self.num_envs, self.num_fingertips), device=self.device)
        self.contact_duration = torch.zeros((self.num_envs, ), device=self.device)
        self.no_contact_duration = torch.zeros((self.num_envs, ), device=self.device)
        self.phase_success_score = torch.zeros((self.num_envs, ), device=self.device)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # delta
        self.delta_obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.delta_fingertip_pos = torch.zeros((self.num_envs, self.num_fingertips), device=self.device)
        
        # delta_value
        self.delta_obj_pos_value = torch.zeros((self.num_envs, ), device=self.device)

        # frame idx
        self.start_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sampled_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # buffers for dof actions
        self.prev_dof_actions = torch.zeros((self.num_envs, self.num_hand_dof), dtype=torch.float, device=self.device)
        self.cur_dof_actions = torch.zeros((self.num_envs, self.num_hand_dof), dtype=torch.float, device=self.device)
        # buffers for external force and torque
        self.prev_forces = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.prev_torques = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # track goal resets
        self.hand_far_apart = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obj_far_apart = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.early_terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # markers
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self.debug_markers = VisualizationMarkers(self.cfg.debug_marker_cfg)
        
        # separate reward logging
        self.logs_dict = dict()

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # global action
        self.is_global = True
        
        self._setup_data()


    def _setup_data(self):
        # Provided code. Do not modify.
        obj_bottom_offset = self.inputs['obj_bottom_offset'].to(self.device)
        obj_reset_pos = torch.zeros((1,3), dtype=torch.float, device=self.device)
        obj_reset_pos[0][2] = self.cfg.table_upper_z + obj_bottom_offset - 0.001
        obj_trans = self.inputs['obj_trans'].to(self.device)
        obj_rot = self.inputs['obj_rot'].to(self.device)
        obj_rot = quat_from_matrix(obj_rot)
        to_center_pos = (- obj_trans[0:1] + obj_reset_pos)

        self.obj_rot_reset[:] = obj_rot[0]
        self.obj_rot_seq = obj_rot
        self.obj_pos_seq = obj_trans + to_center_pos
        self.obj_linvel_seq = self.inputs['obj_vel'].to(self.device)
        self.obj_angvel_seq = self.inputs['obj_angvel'].to(self.device)
        self.obj_linvel_value_seq = torch.norm(self.obj_linvel_seq, p=2, dim=-1)
        self.obj_angvel_value_seq = torch.norm(self.obj_angvel_seq, p=2, dim=-1)
        
        mano_kpts_pos_seq = self.inputs["mano_kpts"][:, self.cfg.MANO_kpts].to(self.device)
        self.mano_kpts_pos_seq = mano_kpts_pos_seq + to_center_pos.unsqueeze(1)
        self.fingertip_pos_seq = self.mano_kpts_pos_seq[:, self.cfg.MANO_fingertips]
        

        self.obj_kpts_pos_seq_offset =  self.mano_kpts_pos_seq - self.obj_pos_seq.unsqueeze(1)
        self.obj_fingertip_pos_seq_offset = self.obj_kpts_pos_seq_offset[:, self.cfg.MANO_fingertips]

        # Use fingertip contact patches as MANO fingertip keypoints.
        seq_len = self.obj_pos_seq.shape[0]
        self.mano_anchor_center_seq = self.mano_kpts_pos_seq[:, self.cfg.MANO_anchors].mean(dim=1)
        self.reference_phase_weights = torch.ones(seq_len, device=self.device)

        dof_lower_limits = self.hand_dof_lower_limits[0] if self.hand_dof_lower_limits.ndim == 2 else self.hand_dof_lower_limits
        dof_upper_limits = self.hand_dof_upper_limits[0] if self.hand_dof_upper_limits.ndim == 2 else self.hand_dof_upper_limits
        self.hand_dof_seq = build_mano_to_shadow_dof_seq(
            self.mano_kpts_pos_seq,
            self.num_hand_dof,
            self.actuated_dof_indices,
            dof_lower_limits,
            dof_upper_limits,
        )
        self.hand_dof_pos_reset[:] = self.hand_dof_seq[0]
        self.hand_rot_reset[:] = self.inputs['R_init'].to(self.device)
        self.hand_pos_reset[:] = (self.inputs['t_init']).to(self.device) + to_center_pos[0]
        # Lift the hand slightly to avoid initial floor contact.
        self.hand_pos_reset[:,2] = self.hand_pos_reset[:,2] + 0.01
        self.hand_pos_reset_base = self.hand_pos_reset.clone()
        self.hand_rot_reset_base = self.hand_rot_reset.clone()
        self.hand_rot_ref_seq = build_anchor_rot_ref_seq(
            self.mano_kpts_pos_seq,
            self.hand_rot_reset_base[0],
        )
    

    def _setup_scene(self):
        # Provided code. Do not modify.

        # add hand, object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # add articulation to scene
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table
        
        self.contact_sensors = [
            self.scene.sensors[f"contact_sensor_{body}"]
            for body in self.cfg.fingertip_body_names
        ]

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # collision group
        stage = omni.usd.get_context().get_stage()
        collisionGroupPaths = [
            "/World/collisionGroup0",
            "/World/collisionGroup1",
            "/World/collisionGroup2",
        ]
        collisionGroupIncludesRel = [None] * 3
        collisionGroupFilteredRels = [None] * 3

        for i in range(3):
            collisionGroup = UsdPhysics.CollisionGroup.Define(stage, collisionGroupPaths[i])
            collisionGroupPrim = collisionGroup.GetPrim()
            collectionAPI = Usd.CollectionAPI.Apply(
                collisionGroupPrim,
                UsdPhysics.Tokens.colliders
            )
            collisionGroupIncludesRel[i] = collectionAPI.CreateIncludesRel()
            collisionGroupFilteredRels[i] = collisionGroup.CreateFilteredGroupsRel()
        
        for i in range(self.num_envs):
            collisionGroupIncludesRel[0].AddTarget(f"/World/envs/env_{i}/Robot")
            collisionGroupIncludesRel[1].AddTarget(f"/World/envs/env_{i}/Object")
            collisionGroupIncludesRel[2].AddTarget(f"/World/envs/env_{i}/table")

        collisionGroupFilteredRels[1].AddTarget(collisionGroupPaths[1])
        collisionGroupFilteredRels[2].AddTarget(collisionGroupPaths[2])


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Provided code. Do not modify.
        self.actions = actions.clone()


    def _apply_action(self) -> None:
        # Provided code. Do not modify.
        pos_offset = self.actions[:, 0:3]
        rot_offset = self.actions[:, 3:9]
        finger_actions = self.actions[:, 9:]
        
        R_offset= rotation_6d_to_matrix(rot_offset)

        # Convert actions into forces and torques
        forces = pos_offset * self.cfg.action_dt * self.cfg.K_pos
        torques = matrix_to_axis_angle(R_offset) * self.cfg.action_dt * self.cfg.K_rot
        forces = (1.0 - self.cfg.global_moving_average) * self.prev_forces + self.cfg.global_moving_average * forces
        torques = (1.0 - self.cfg.global_moving_average) * self.prev_torques + self.cfg.global_moving_average * torques
        with torch.no_grad():
            self.prev_forces = forces.detach().clone()
            self.prev_torques = torques.detach().clone()
        full_forces = torch.zeros((self.num_envs, self.hand.num_bodies, 3), device=self.device)
        full_torques = torch.zeros((self.num_envs, self.hand.num_bodies, 3), device=self.device)

        # Apply forces and torques only on the root(palm)
        full_forces[:, self.root_body[0], :] = forces
        full_torques[:, self.root_body[0], :] = torques
        self.hand.set_external_force_and_torque(
            full_forces,
            full_torques,
            is_global=True,
        )

        
        # Scale DoF and Smooth finger actions
        self.cur_dof_actions[:, self.actuated_dof_indices] = scale(
            finger_actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        
        self.cur_dof_actions[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_dof_actions[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_dof_actions[:, self.actuated_dof_indices]
        )
        
        self.cur_dof_actions[:, self.actuated_dof_indices] = saturate(
            self.cur_dof_actions[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        
        self.prev_dof_actions[:, self.actuated_dof_indices] = self.cur_dof_actions[:, self.actuated_dof_indices]
        # Position control for fingers
        self.hand.set_joint_position_target(
            self.cur_dof_actions[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices
        )


    def _get_observations(self) -> dict:
        # Provided code. Do not modify.
        obs = self.compute_full_observations()
        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        curriculum_progress = torch.full(
            (self.num_envs,),
            float(
                np.clip(
                    getattr(self, "common_step_counter", 0) / max(1, self.cfg.reward_curriculum_steps),
                    0.0,
                    1.0,
                )
            ),
            device=self.device,
        )
        early_episode_gate = torch.clamp(
            1.0 - self.episode_length_buf.float() / max(1.0, self.cfg.early_episode_tracking_frames),
            0.0,
            1.0,
        )
        (
            total_reward,
            logs_dict,
        ) = compute_rewards(
            # TODO: Pass all tensors and scalar weights required by compute_rewards().
            self.actions,
            self.hand_kpt_err,
            self.hand_pos_err,
            self.hand_anchor_err,
            self.hand_obj_offset_err,
            self.anchor_obj_offset_err,
            self.hand_dof_err,
            self.hand_rot_err,
            self.finger_shape_err,
            self.finger_topology_err,
            self.fingertip_err,
            self.fingertip_obj_proximity_err,
            self.fingertip_obj_topk_proximity_err,
            self.fingertip_obj_offset_err,
            self.contact_force,
            self.projected_contact_force,
            self.max_contact_force,
            self.contact_active,
            self.num_contact_fingers,
            self.ref_fingertip_contact_weights,
            self.proximity_gate,
            self.contact_sustain,
            self.no_contact_sustain,
            self.start_frame_idx,
            self.obj_pos_err,
            self.obj_delta_err,
            self.obj_rot_err,
            self.obj_linvel_error,
            self.obj_angvel_error,
            self.obj_future_dir_reward,
            curriculum_progress,
            early_episode_gate,
            self.cfg.hand_reward_weight,
            self.cfg.hand_pos_reward_weight,
            self.cfg.hand_anchor_reward_weight,
            self.cfg.hand_obj_offset_reward_weight,
            self.cfg.anchor_obj_offset_reward_weight,
            self.cfg.hand_dof_reward_weight,
            self.cfg.hand_rot_reward_weight,
            self.cfg.finger_shape_reward_weight,
            self.cfg.finger_topology_reward_weight,
            self.cfg.fingertip_reward_weight,
            self.cfg.fingertip_obj_proximity_reward_weight,
            self.cfg.fingertip_obj_offset_reward_weight,
            self.cfg.contact_reward_weight,
            self.cfg.obj_pos_reward_weight,
            self.cfg.obj_rot_reward_weight,
            self.cfg.obj_vel_reward_weight,
            self.cfg.action_penalty_scale,
            self.cfg.hand_reward_scale,
            self.cfg.hand_pos_reward_scale,
            self.cfg.hand_anchor_reward_scale,
            self.cfg.hand_obj_offset_reward_scale,
            self.cfg.anchor_obj_offset_reward_scale,
            self.cfg.hand_dof_reward_scale,
            self.cfg.hand_rot_reward_scale,
            self.cfg.finger_shape_reward_scale,
            self.cfg.finger_topology_reward_scale,
            self.cfg.finger_shape_contact_decay,
            self.cfg.finger_topology_contact_decay,
            self.cfg.anchor_rotation_gate_scale,
            self.cfg.fingertip_reward_scale,
            self.cfg.fingertip_obj_proximity_reward_scale,
            self.cfg.fingertip_obj_offset_reward_scale,
            self.cfg.object_reward_gate_base,
            self.cfg.contact_force_reward_weight,
            self.cfg.contact_count_reward_weight,
            self.cfg.contact_sustain_reward_weight,
            self.cfg.stable_grasp_reward_weight,
            self.cfg.transport_support_reward_weight,
            self.cfg.obj_future_dir_reward_weight,
            self.cfg.grasped_hand_ref_reward_weight,
            self.cfg.early_imitation_reward_bonus,
            self.cfg.early_episode_tracking_bonus,
            self.cfg.early_lag_penalty_weight,
            self.cfg.approach_imitation_bonus,
            self.cfg.grasp_object_bonus,
            self.cfg.manipulation_task_bonus,
            self.cfg.manipulation_imitation_bonus,
            self.cfg.successful_grasp_dof_bonus_weight,
            self.cfg.pre_contact_pose_bonus_weight,
            self.cfg.no_contact_mano_imitation_floor,
            self.cfg.object_relative_reward_base,
            self.cfg.mid_object_relative_reward_bonus,
            self.cfg.late_task_reward_bonus,
            self.cfg.anchor_object_gate_floor,
            self.cfg.no_contact_late_reward_floor,
            self.cfg.no_grasp_rotation_penalty_weight,
            self.cfg.target_contact_fingers,
            self.cfg.contact_reward_max_force,
            self.cfg.force_dominance_limit,
            self.cfg.obj_pos_reward_scale,
            self.cfg.obj_delta_reward_scale,
            self.cfg.obj_rot_reward_scale,
            self.cfg.obj_vel_reward_scale,
            self.cfg.obj_angvel_reward_scale,
        )

        for key, value in logs_dict.items():
            if key not in self.logs_dict:
                self.logs_dict[key] = value.detach()
            else:
                self.logs_dict[key] += value.detach()

        if "log" not in self.extras:
            self.extras["log"] = dict()

        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        self.time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        early_terminate = self.early_terminate if self.termination else torch.zeros_like(self.early_terminate, device=self.device)
        return early_terminate, self.time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        self._update_reference_phase_weights(env_ids)
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)
        self._sample_reference_phase(env_ids)
        # Reset object
        self._reset_object(env_ids)
        # Reset hand
        self._reset_hand(env_ids)

        for key, value in self.logs_dict.items():
            self.extras["log"][key] = value.mean()
        self.logs_dict = dict()
        
        self.successes[env_ids] = 0
        self.contact_duration[env_ids] = 0
        self.no_contact_duration[env_ids] = 0
        self.phase_success_score[env_ids] = 0
        self._compute_intermediate_values()


    def _sample_reference_phase(self, env_ids):
        if self.play or not self.cfg.random_reference_phase_sampling:
            self.start_frame_idx[env_ids] = 0
            self.sampled_frame_idx[env_ids] = 0
            self.episode_length_buf[env_ids] = 0
            return

        max_start = max(0, self.max_episode_length - int(self.cfg.reference_phase_min_remaining_steps) - 1)
        num_envs = len(env_ids)
        zero_frames = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        uniform_frames = torch.randint(0, max_start + 1, (num_envs,), device=self.device)
        weights = self.reference_phase_weights[: max_start + 1].clamp_min(1.0e-6)
        if self.cfg.success_biased_phase_sampling:
            biased_frames = torch.multinomial(weights, num_envs, replacement=True)
        else:
            biased_frames = uniform_frames

        phase_sample = torch.rand(num_envs, device=self.device)
        frame0_cutoff = self.cfg.reference_phase_frame0_ratio
        uniform_cutoff = frame0_cutoff + self.cfg.reference_phase_uniform_ratio
        sampled_frames = torch.where(
            phase_sample < frame0_cutoff,
            zero_frames,
            torch.where(phase_sample < uniform_cutoff, uniform_frames, biased_frames),
        ).long()

        self.start_frame_idx[env_ids] = sampled_frames
        self.sampled_frame_idx[env_ids] = sampled_frames
        self.episode_length_buf[env_ids] = sampled_frames


    def _update_reference_phase_weights(self, env_ids):
        if not hasattr(self, "reference_phase_weights"):
            return
        if self.play or not self.cfg.success_biased_phase_sampling:
            return

        scores = self.phase_success_score[env_ids].detach().clamp_min(0.0)
        active = scores > 0.0
        if not torch.any(active):
            return

        frames = self.start_frame_idx[env_ids][active].long()
        scores = scores[active]
        max_index = self.reference_phase_weights.shape[0] - 1
        radius = int(self.cfg.success_phase_spread)
        self.reference_phase_weights.mul_(self.cfg.success_phase_weight_decay).clamp_(min=1.0)
        for offset in range(-radius, radius + 1):
            idx = torch.clamp(frames + offset, 0, max_index)
            self.reference_phase_weights.index_add_(0, idx, scores * self.cfg.success_phase_weight_gain)


    def _set_object_state(self, pos, rot, env_ids, vel=None):
        default_states = self.object.data.default_root_state[env_ids].clone()
        default_states[:, :3] = pos + self.scene.env_origins[env_ids]
        default_states[:, 3:7] = rot

        if vel is not None:
            default_states[:, 7:13] = vel
        
        self.object.write_root_state_to_sim(default_states, env_ids=env_ids)

        self.obj_pos[env_ids] = self.obj_pos_reset[env_ids]
        self.obj_rot[env_ids] = self.obj_rot_reset[env_ids]


    def _reset_object(self, env_ids):
        frame_ids = self.start_frame_idx[env_ids].long()
        self.obj_pos_reset[env_ids] = self.obj_pos_seq[frame_ids]
        self.obj_rot_reset[env_ids] = self.obj_rot_seq[frame_ids]
        obj_vel = torch.cat((self.obj_linvel_seq[frame_ids], self.obj_angvel_seq[frame_ids]), dim=-1)
        self._set_object_state(self.obj_pos_reset[env_ids], self.obj_rot_reset[env_ids], env_ids, obj_vel)


    def _set_hand_state(self, pos, rot, dof_pos, dof_vel, root_vel, dof_target, ext_force, ext_torque, env_ids):
        hand_default_state = self.hand.data.default_root_state.clone()
        hand_default_state[env_ids, 0:3] = pos + self.scene.env_origins[env_ids]
        hand_default_state[env_ids, 3:7] = rot
        hand_default_state[env_ids, 7:13] = root_vel

        self.hand.write_root_pose_to_sim(hand_default_state[env_ids, :7], env_ids=env_ids)
        self.hand.write_root_velocity_to_sim(hand_default_state[env_ids, 7:13], env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.hand.set_joint_position_target(dof_target[:, self.actuated_dof_indices], self.actuated_dof_indices, env_ids=env_ids)
        self.hand.set_external_force_and_torque(ext_force, ext_torque, env_ids=env_ids, is_global=self.is_global)

        self.prev_dof_actions[env_ids] = dof_target.clone()
        self.cur_dof_actions[env_ids] = dof_target.clone()
        self.prev_forces[env_ids] = ext_force[:, self.root_body[0], :].clone()
        self.prev_torques[env_ids] = ext_torque[:, self.root_body[0], :].clone()
        
        self.hand_pos[env_ids] = self.hand_pos_reset[env_ids]
        self.hand_rot[env_ids] = self.hand_rot_reset[env_ids]


    def _reset_hand(self, env_ids):
        frame_ids = self.start_frame_idx[env_ids].long()
        hand_anchor_delta = self.mano_anchor_center_seq[frame_ids] - self.mano_anchor_center_seq[0]
        self.hand_pos_reset[env_ids] = self.hand_pos_reset_base[env_ids] + hand_anchor_delta
        self.hand_rot_reset[env_ids] = self.hand_rot_ref_seq[frame_ids]

        dof_pos = self.hand_dof_seq[frame_ids]
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        root_vel = torch.zeros_like(self.hand.data.default_root_state[env_ids, 7:13])

        hand_global_force = torch.zeros((len(env_ids), self.hand.num_bodies, 3), device=self.device)
        hand_global_torque = torch.zeros((len(env_ids), self.hand.num_bodies, 3), device=self.device)

        self._set_hand_state(self.hand_pos_reset[env_ids], self.hand_rot_reset[env_ids], dof_pos, dof_vel, root_vel, dof_pos, hand_global_force, hand_global_torque, env_ids)


    def _collect_target(self):
        t = self.episode_length_buf
        self.t = t
        t_next = torch.clamp(t + 1, max=(self.max_episode_length-1))
        
        # current ref
        self.obj_pos_ref = self.obj_pos_seq[t]
        self.obj_rot_ref = self.obj_rot_seq[t]
        self.obj_linvel_ref = self.obj_linvel_seq[t]
        self.obj_angvel_ref = self.obj_angvel_seq[t]
        self.obj_linvel_value_ref = self.obj_linvel_value_seq[t]
        self.obj_angvel_value_ref = self.obj_angvel_value_seq[t]

        self.fingertip_pos_ref = self.fingertip_pos_seq[t]
        self.mano_kpts_pos_ref = self.mano_kpts_pos_seq[t]
        self.hand_dof_ref = self.hand_dof_seq[t]
        self.obj_fingertip_pos_ref_offset = self.obj_fingertip_pos_seq_offset[t]
        hand_anchor_delta = self.mano_anchor_center_seq[t] - self.mano_anchor_center_seq[0]
        self.hand_pos_ref = self.hand_pos_reset_base + hand_anchor_delta
        self.hand_rot_ref = self.hand_rot_ref_seq[t]
        self.hand_obj_ref_offset = self.hand_pos_ref - self.obj_pos_ref
        self.anchor_obj_ref_offset = self.mano_anchor_center_seq[t] - self.obj_pos_ref
        
        # next ref
        self.obj_pos_next = self.obj_pos_seq[t_next]
        self.obj_rot_next = self.obj_rot_seq[t_next]
        self.obj_linvel_next = self.obj_linvel_seq[t_next]
        self.obj_angvel_next = self.obj_angvel_seq[t_next]
        self.obj_linvel_value_next = self.obj_linvel_value_seq[t_next]
        self.obj_angvel_value_next = self.obj_angvel_value_seq[t_next]

        self.hand_dof_next = self.hand_dof_seq[t_next]
        self.fingertip_pos_next = self.fingertip_pos_seq[t_next]
        self.mano_kpts_pos_next = self.mano_kpts_pos_seq[t_next]


    def _collect_state(self):
        # data for object
        object_state = self.object.data.root_state_w
        self.obj_pos = object_state[:,:3] - self.scene.env_origins
        self.obj_rot = object_state[:,3:7] 
        self.obj_linvel = object_state[:,7:10]
        self.obj_angvel = object_state[:,10:13]

        # data for hand
        hand_state = self.hand.data.root_state_w
        self.hand_pos = hand_state[:, :3] - self.scene.env_origins
        self.hand_rot = hand_state[:, 3:7]
        self.hand_linvel = hand_state[:,7:10]
        self.hand_angvel = hand_state[:,10:13]
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for handbodies
        body_state = self.hand.data.body_state_w[:, self.hand_bodies]
        hand_bodies_pos = body_state[:, :, :3]
        self.hand_bodies_pos = hand_bodies_pos - self.scene.env_origins.unsqueeze(1)
        self.hand_bodies_rot = body_state[:, :, 3:7]
        self.hand_bodies_linvel = body_state[:, :, 7:10]
        self.hand_bodies_angvel = body_state[:, :, 10:13]

        # data for fingertips
        fingertip_pos = self.hand_bodies_pos[:, self.fingertip_bodies]
        self.fingertip_rot = self.hand_bodies_rot[:, self.fingertip_bodies]
        self.fingertip_linvel = self.hand_bodies_linvel[:, self.fingertip_bodies]
        self.fingertip_angvel = self.hand_bodies_angvel[:, self.fingertip_bodies]

        # normal, axis
        self.normal = quat_apply(self.fingertip_rot, self.fingertip_normal)
        offset = quat_apply(self.fingertip_rot, self.fingertip_offset)
        # Use fingertip contact patches as MANO fingertip keypoints.
        self.hand_kpts_pos[:, self.cfg.MANO_kpts_except_fingertips] = self.hand_bodies_pos[:, self.cfg.body_to_kpts_except_fingertips]
        self.hand_kpts_pos[:, self.cfg.MANO_fingertips] = fingertip_pos + offset

        self.fingertip_pos = self.hand_kpts_pos[:, self.cfg.MANO_fingertips]
        
        # data for fingertip sensors
        for i in range(self.num_fingertips):
            force = self.contact_sensors[i].data.force_matrix_w
            self.fingertip_contact_forces[:, i] = force[:, 0, 0]
        self.fingertip_contact_force_norm = torch.norm(self.fingertip_contact_forces, p=2, dim=-1)
        self.fingertip_contact_forces_buf[:, 0] = torch.clamp_min((self.fingertip_contact_forces * (-self.normal)).sum(dim=-1), 0)
        self.fingertip_contact_forces_buf[:, 1] = self.fingertip_contact_force_norm



    def _compute_intermediate_values(self):
        self._collect_target()
        self._collect_state()

        # TODO: Compute intermediate values used by observations and rewards.
        self.hand_kpt_error = self.hand_kpts_pos - self.mano_kpts_pos_ref
        self.hand_anchor_error = self.hand_kpts_pos[:, self.cfg.MANO_anchors] - self.mano_kpts_pos_ref[:, self.cfg.MANO_anchors]
        self.fingertip_error = self.fingertip_pos - self.fingertip_pos_ref
        self.hand_anchor_center = self.hand_kpts_pos[:, self.cfg.MANO_anchors].mean(dim=1)

        self.hand_kpt_err = torch.norm(self.hand_kpt_error, p=2, dim=-1).mean(dim=-1)
        self.hand_pos_err = torch.norm(self.hand_pos - self.hand_pos_ref, p=2, dim=-1)
        self.hand_anchor_err = torch.norm(self.hand_anchor_error, p=2, dim=-1).mean(dim=-1)
        self.hand_dof_err = torch.abs(self.hand_dof_pos - self.hand_dof_ref).mean(dim=-1)
        self.fingertip_err = torch.norm(self.fingertip_error, p=2, dim=-1).mean(dim=-1)
        hand_finger_kpts = self.hand_kpts_pos[:, self.finger_joint_groups]
        ref_finger_kpts = self.mano_kpts_pos_ref[:, self.finger_joint_groups]
        finger_joint_err = torch.norm(hand_finger_kpts - ref_finger_kpts, p=2, dim=-1)
        per_finger_pos_err = finger_joint_err.mean(dim=-1)
        hand_finger_dirs = F.normalize(hand_finger_kpts[:, :, 1:] - hand_finger_kpts[:, :, :-1], dim=-1, eps=1.0e-6)
        ref_finger_dirs = F.normalize(ref_finger_kpts[:, :, 1:] - ref_finger_kpts[:, :, :-1], dim=-1, eps=1.0e-6)
        finger_dir_err = torch.norm(hand_finger_dirs - ref_finger_dirs, p=2, dim=-1).mean(dim=-1)
        finger_pos_err = 0.7 * per_finger_pos_err.mean(dim=-1) + 0.3 * torch.max(per_finger_pos_err, dim=-1).values
        finger_rot_like_err = 0.7 * finger_dir_err.mean(dim=-1) + 0.3 * torch.max(finger_dir_err, dim=-1).values
        self.finger_shape_err = finger_pos_err + self.cfg.finger_direction_error_weight * finger_rot_like_err
        hand_topology_kpts = self.hand_kpts_pos[:, self.topology_joint_indices]
        ref_topology_kpts = self.mano_kpts_pos_ref[:, self.topology_joint_indices]
        hand_topology_dist = torch.cdist(hand_topology_kpts, hand_topology_kpts, p=2)
        ref_topology_dist = torch.cdist(ref_topology_kpts, ref_topology_kpts, p=2)
        hand_tip_dist = torch.cdist(self.fingertip_pos, self.fingertip_pos, p=2)
        ref_tip_dist = torch.cdist(self.fingertip_pos_ref, self.fingertip_pos_ref, p=2)
        topology_joint_err = torch.abs(hand_topology_dist - ref_topology_dist).mean(dim=(1, 2))
        topology_tip_err = torch.abs(hand_tip_dist - ref_tip_dist).mean(dim=(1, 2))
        self.finger_topology_err = 0.5 * topology_joint_err + 0.5 * topology_tip_err
        self.hand_obj_offset_err = torch.norm(
            (self.hand_pos - self.obj_pos) - self.hand_obj_ref_offset,
            p=2,
            dim=-1,
        )
        self.anchor_obj_offset_err = torch.norm(
            (self.hand_anchor_center - self.obj_pos) - self.anchor_obj_ref_offset,
            p=2,
            dim=-1,
        )

        self.hand_rot_error_quat = quat_mul(self.hand_rot, quat_conjugate(self.hand_rot_ref))
        hand_rot_vec_norm = torch.norm(self.hand_rot_error_quat[:, 1:4], p=2, dim=-1)
        hand_rot_w = torch.abs(self.hand_rot_error_quat[:, 0])
        self.hand_rot_err = 2.0 * torch.atan2(hand_rot_vec_norm, hand_rot_w)

        self.obj_pos_error = self.obj_pos - self.obj_pos_ref
        self.obj_pos_err = torch.norm(self.obj_pos_error, p=2, dim=-1)
        self.obj_delta_error = (self.obj_pos - self.obj_pos_reset) - (self.obj_pos_ref - self.obj_pos_reset)
        self.obj_delta_err = torch.norm(self.obj_delta_error, p=2, dim=-1)

        self.obj_rot_error_quat = quat_mul(self.obj_rot, quat_conjugate(self.obj_rot_ref))
        obj_rot_vec_norm = torch.norm(self.obj_rot_error_quat[:, 1:4], p=2, dim=-1)
        obj_rot_w = torch.abs(self.obj_rot_error_quat[:, 0])
        self.obj_rot_err = 2.0 * torch.atan2(obj_rot_vec_norm, obj_rot_w)

        self.obj_linvel_error = self.obj_linvel - self.obj_linvel_ref
        self.obj_angvel_error = self.obj_angvel - self.obj_angvel_ref

        self.fingertip_to_obj = self.fingertip_pos - self.obj_pos.unsqueeze(1)
        self.fingertip_obj_dist = torch.norm(self.fingertip_to_obj, p=2, dim=-1)
        self.fingertip_obj_proximity_err = self.fingertip_obj_dist.mean(dim=-1)
        topk_count = min(self.cfg.contact_topk_fingers, self.num_fingertips)
        self.fingertip_obj_topk_proximity_err = torch.topk(
            self.fingertip_obj_dist,
            k=topk_count,
            dim=-1,
            largest=False,
        ).values.mean(dim=-1)
        self.fingertip_obj_offset_error = self.fingertip_to_obj - self.obj_fingertip_pos_ref_offset
        self.fingertip_obj_offset_err = torch.norm(self.fingertip_obj_offset_error, p=2, dim=-1).mean(dim=-1)
        ref_fingertip_obj_dist = torch.norm(self.obj_fingertip_pos_ref_offset, p=2, dim=-1)
        ref_dist_min = ref_fingertip_obj_dist.min(dim=-1, keepdim=True).values
        self.ref_fingertip_contact_weights = torch.exp(
            -self.cfg.reference_contact_proximity_scale * torch.clamp_min(ref_fingertip_obj_dist - ref_dist_min, 0.0)
        )
        self.projected_contact_force = self.fingertip_contact_forces_buf[:, 0, :].sum(dim=-1)
        self.contact_force = self.fingertip_contact_force_norm.sum(dim=-1)
        self.max_contact_force = self.fingertip_contact_force_norm.max(dim=-1).values
        self.contact_active = (self.fingertip_contact_force_norm > self.cfg.contact_force_threshold).float()
        self.fingertip_contact_forces_buf[:, 2] = self.contact_active
        self.num_contact_fingers = self.contact_active.sum(dim=-1)
        self.contact_duration = torch.where(
            self.num_contact_fingers > 0,
            self.contact_duration + 1.0,
            torch.zeros_like(self.contact_duration),
        )
        self.no_contact_duration = torch.where(
            self.num_contact_fingers > 0,
            torch.zeros_like(self.no_contact_duration),
            self.no_contact_duration + 1.0,
        )
        self.contact_sustain = torch.clamp(
            self.contact_duration / (self.cfg.contact_sustain_target_steps + 1.0e-6),
            0.0,
            1.0,
        )
        self.no_contact_sustain = torch.clamp(
            self.no_contact_duration / (self.cfg.no_contact_grace_steps + 1.0e-6),
            0.0,
            1.0,
        )
        self.proximity_gate = torch.exp(-self.cfg.proximity_gate_scale * self.fingertip_obj_topk_proximity_err)
        phase_obj_pos_reward = torch.exp(-self.cfg.obj_pos_reward_scale * self.obj_pos_err)
        phase_obj_rot_reward = torch.exp(-self.cfg.obj_rot_reward_scale * self.obj_rot_err)
        phase_fingertip_obj_offset_reward = torch.exp(
            -self.cfg.fingertip_obj_offset_reward_scale * self.fingertip_obj_offset_err
        )
        phase_topology_reward = torch.exp(-self.cfg.finger_topology_reward_scale * self.finger_topology_err)
        phase_thumb_ref = torch.clamp(self.ref_fingertip_contact_weights[:, 0], 0.0, 1.0)
        phase_thumb_contact = self.contact_active[:, 0]
        phase_thumb_gate = (1.0 - phase_thumb_ref) + phase_thumb_ref * phase_thumb_contact
        phase_non_thumb_ref = self.ref_fingertip_contact_weights[:, 1:]
        phase_non_thumb_target = phase_non_thumb_ref.sum(dim=-1)
        phase_non_thumb_presence = torch.clamp(phase_non_thumb_target, 0.0, 1.0)
        phase_non_thumb_contact = (self.contact_active[:, 1:] * phase_non_thumb_ref).sum(dim=-1)
        phase_non_thumb_gate = (1.0 - phase_non_thumb_presence) + phase_non_thumb_presence * torch.clamp(
            phase_non_thumb_contact / (phase_non_thumb_target + 1.0e-6),
            0.0,
            1.0,
        )
        phase_force_dominance = self.max_contact_force / (self.contact_force + 1.0e-6)
        phase_force_balance = torch.clamp(
            (self.cfg.force_dominance_limit - phase_force_dominance)
            / (self.cfg.force_dominance_limit - 1.0 / self.cfg.target_contact_fingers + 1.0e-6),
            0.0,
            1.0,
        )
        phase_stable_grasp_gate = (
            phase_thumb_gate
            * phase_non_thumb_gate
            * phase_force_balance
            * torch.clamp(self.contact_sustain, 0.0, 1.0)
        )
        phase_transport_gate = 0.20 * torch.clamp(self.contact_sustain, 0.0, 1.0) + 0.80 * phase_stable_grasp_gate
        phase_reference_success = phase_transport_gate * phase_topology_reward * phase_fingertip_obj_offset_reward * (
            0.5 * phase_obj_pos_reward + 0.5 * phase_obj_rot_reward
        )
        self.phase_success_score = torch.maximum(
            self.phase_success_score,
            torch.clamp(phase_reference_success, 0.0, 1.0),
        )

        self.palm_to_obj = self.hand_pos - self.obj_pos
        self.palm_obj_dist = torch.norm(self.palm_to_obj, p=2, dim=-1)
        obj_future_delta = self.obj_pos_next - self.obj_pos
        obj_future_dir = F.normalize(obj_future_delta, dim=-1, eps=1.0e-6)
        obj_linvel_dir = F.normalize(self.obj_linvel, dim=-1, eps=1.0e-6)
        obj_future_dist_gate = torch.clamp(
            torch.norm(obj_future_delta, p=2, dim=-1) / (self.cfg.obj_future_dir_min_distance + 1.0e-6),
            0.0,
            1.0,
        )
        self.obj_future_dir_reward = torch.clamp((obj_linvel_dir * obj_future_dir).sum(dim=-1), 0.0, 1.0) * obj_future_dist_gate

        self.delta_obj_pos = self.obj_pos_error
        self.delta_obj_pos_value = self.obj_pos_err
        self.delta_fingertip_pos = torch.norm(self.fingertip_error, p=2, dim=-1)

        self.hand_far_apart = self.hand_kpt_err > self.cfg.hand_terminate_threshold
        self.obj_far_apart = self.obj_pos_err > self.cfg.obj_terminate_threshold
        self.obj_rot_far_apart = torch.logical_and(
            self.episode_length_buf > self.cfg.obj_rot_terminate_after_steps,
            self.obj_rot_err > self.cfg.obj_rot_terminate_threshold,
        )
        self.no_grasp_failure = torch.logical_and(
            self.episode_length_buf > self.cfg.no_grasp_terminate_after_steps,
            self.no_contact_duration > self.cfg.no_grasp_terminate_grace_steps,
        )
        self.early_terminate = torch.logical_or(
            torch.logical_or(self.hand_far_apart, self.obj_far_apart),
            torch.logical_or(self.obj_rot_far_apart, self.no_grasp_failure),
        )
        self._compute_lookahead_observations()

        if not self.play:
            # Point visualization for debugging; you may change which points are shown.
            debug_vis1 = self.mano_kpts_pos_ref[:, self.cfg.MANO_fingertips] + self.scene.env_origins.unsqueeze(1)
            self.goal_markers.visualize(debug_vis1.view(-1,3))
            debug_vis2 = self.hand_kpts_pos[:, self.cfg.MANO_fingertips] + self.scene.env_origins.unsqueeze(1)
            self.debug_markers.visualize(debug_vis2.view(-1,3))


    def _compute_lookahead_observations(self):
        lookahead_parts = []
        object_lookahead_parts = []
        for step in self.cfg.lookahead_reference_steps:
            t_future = torch.clamp(
                self.episode_length_buf + int(step),
                max=(self.max_episode_length - 1),
            )
            hand_anchor_delta = self.mano_anchor_center_seq[t_future] - self.mano_anchor_center_seq[0]
            hand_pos_future = self.hand_pos_reset_base + hand_anchor_delta
            fingertip_pos_future = self.fingertip_pos_seq[t_future]
            fingertip_obj_offset_future = self.obj_fingertip_pos_seq_offset[t_future]
            obj_pos_future = self.obj_pos_seq[t_future]
            obj_rot_future = self.obj_rot_seq[t_future]
            obj_rot_future_error_quat = quat_mul(self.obj_rot, quat_conjugate(obj_rot_future))

            lookahead_parts.extend(
                (
                    hand_pos_future - self.hand_pos,
                    (fingertip_pos_future - self.fingertip_pos).reshape(self.num_envs, -1),
                    (fingertip_obj_offset_future - self.fingertip_to_obj).reshape(self.num_envs, -1),
                )
            )
            object_lookahead_parts.extend(
                (
                    obj_pos_future - self.obj_pos,
                    quat_to_6d(obj_rot_future_error_quat),
                )
            )

        self.lookahead_ref_obs = torch.cat(lookahead_parts, dim=-1)
        self.object_lookahead_ref_obs = torch.cat(object_lookahead_parts, dim=-1)


    def compute_full_observations(self):
        obs_parts = (
            unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
            self.hand_dof_vel * self.cfg.vel_obs_scale,

            self.hand_pos - self.obj_pos,
            quat_to_6d(self.hand_rot),
            self.hand_linvel * self.cfg.vel_obs_scale,
            self.hand_angvel * self.cfg.vel_obs_scale,

            self.obj_pos,
            quat_to_6d(self.obj_rot),
            self.obj_linvel * self.cfg.vel_obs_scale,
            self.obj_angvel * self.cfg.vel_obs_scale,

            self.hand_kpt_error.reshape(self.num_envs, -1),
            self.fingertip_to_obj.reshape(self.num_envs, -1),
            self.lookahead_ref_obs,
            self.object_lookahead_ref_obs,

            self.obj_pos_error,
            quat_to_6d(self.obj_rot_error_quat),

            self.fingertip_contact_forces_buf.reshape(self.num_envs, -1),
        )
        obs = torch.cat(
            # TODO: Build the policy observation vector.
            # Its final dimension must match cfg.observation_space.
            obs_parts,
            dim=-1,
        )
        return obs
    

@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def compute_rewards(
    # TODO: Define the reward function inputs with TorchScript-compatible types.
    # EX) actions: torch.Tensor,
    actions: torch.Tensor,
    hand_kpt_err: torch.Tensor,
    hand_pos_err: torch.Tensor,
    hand_anchor_err: torch.Tensor,
    hand_obj_offset_err: torch.Tensor,
    anchor_obj_offset_err: torch.Tensor,
    hand_dof_err: torch.Tensor,
    hand_rot_err: torch.Tensor,
    finger_shape_err: torch.Tensor,
    finger_topology_err: torch.Tensor,
    fingertip_err: torch.Tensor,
    fingertip_obj_proximity_err: torch.Tensor,
    fingertip_obj_topk_proximity_err: torch.Tensor,
    fingertip_obj_offset_err: torch.Tensor,
    contact_force: torch.Tensor,
    projected_contact_force: torch.Tensor,
    max_contact_force: torch.Tensor,
    contact_active: torch.Tensor,
    num_contact_fingers: torch.Tensor,
    ref_fingertip_contact_weights: torch.Tensor,
    proximity_gate: torch.Tensor,
    contact_sustain: torch.Tensor,
    no_contact_sustain: torch.Tensor,
    start_frame_idx: torch.Tensor,
    obj_pos_err: torch.Tensor,
    obj_delta_err: torch.Tensor,
    obj_rot_err: torch.Tensor,
    obj_linvel_error: torch.Tensor,
    obj_angvel_error: torch.Tensor,
    obj_future_dir_reward: torch.Tensor,
    curriculum_progress: torch.Tensor,
    early_episode_gate: torch.Tensor,
    hand_weight: float,
    hand_pos_weight: float,
    hand_anchor_weight: float,
    hand_obj_offset_weight: float,
    anchor_obj_offset_weight: float,
    hand_dof_weight: float,
    hand_rot_weight: float,
    finger_shape_weight: float,
    finger_topology_weight: float,
    fingertip_weight: float,
    fingertip_obj_proximity_weight: float,
    fingertip_obj_offset_weight: float,
    contact_weight: float,
    obj_pos_weight: float,
    obj_rot_weight: float,
    obj_vel_weight: float,
    action_penalty_scale: float,
    hand_reward_scale: float,
    hand_pos_reward_scale: float,
    hand_anchor_reward_scale: float,
    hand_obj_offset_reward_scale: float,
    anchor_obj_offset_reward_scale: float,
    hand_dof_reward_scale: float,
    hand_rot_reward_scale: float,
    finger_shape_reward_scale: float,
    finger_topology_reward_scale: float,
    finger_shape_contact_decay: float,
    finger_topology_contact_decay: float,
    anchor_rotation_gate_scale: float,
    fingertip_reward_scale: float,
    fingertip_obj_proximity_reward_scale: float,
    fingertip_obj_offset_reward_scale: float,
    object_reward_gate_base: float,
    contact_force_reward_weight: float,
    contact_count_reward_weight: float,
    contact_sustain_reward_weight: float,
    stable_grasp_reward_weight: float,
    transport_support_reward_weight: float,
    obj_future_dir_reward_weight: float,
    grasped_hand_ref_reward_weight: float,
    early_imitation_reward_bonus: float,
    early_episode_tracking_bonus: float,
    early_lag_penalty_weight: float,
    approach_imitation_bonus: float,
    grasp_object_bonus: float,
    manipulation_task_bonus: float,
    manipulation_imitation_bonus: float,
    successful_grasp_dof_bonus_weight: float,
    pre_contact_pose_bonus_weight: float,
    no_contact_mano_imitation_floor: float,
    object_relative_reward_base: float,
    mid_object_relative_reward_bonus: float,
    late_task_reward_bonus: float,
    anchor_object_gate_floor: float,
    no_contact_late_reward_floor: float,
    no_grasp_rotation_penalty_weight: float,
    target_contact_fingers: float,
    contact_reward_max_force: float,
    force_dominance_limit: float,
    obj_pos_reward_scale: float,
    obj_delta_reward_scale: float,
    obj_rot_reward_scale: float,
    obj_vel_reward_scale: float,
    obj_angvel_reward_scale: float,
):
    # TODO: Compute rewards and combine them into the final reward.
    obj_linvel_err = torch.norm(obj_linvel_error, p=2, dim=-1)
    obj_angvel_err = torch.norm(obj_angvel_error, p=2, dim=-1)
    obj_vel_err = obj_linvel_err + obj_angvel_reward_scale * obj_angvel_err
    early_curriculum = 1.0 - curriculum_progress
    mid_curriculum = 1.0 - torch.abs(2.0 * curriculum_progress - 1.0)
    approach_phase = early_episode_gate
    grasp_phase = (1.0 - approach_phase) * (1.0 - contact_sustain)
    manipulation_phase = contact_sustain
    approach_imitation_scale = 1.0 + approach_imitation_bonus * approach_phase
    contact_or_proximity = torch.maximum(contact_sustain, proximity_gate)
    no_contact_mano_gate = approach_phase + (1.0 - approach_phase) * (
        no_contact_mano_imitation_floor
        + (1.0 - no_contact_mano_imitation_floor) * contact_or_proximity
    )
    position_imitation_scale = (
        1.0
        + 0.5 * early_imitation_reward_bonus * early_curriculum
        + early_episode_tracking_bonus * approach_phase
    ) * approach_imitation_scale
    pose_imitation_scale = (
        1.0
        + early_imitation_reward_bonus * early_curriculum
        + early_episode_tracking_bonus * approach_phase
    ) * approach_imitation_scale
    mano_regrasp_scale = 1.0 + manipulation_imitation_bonus * manipulation_phase
    object_relative_scale = object_relative_reward_base + mid_object_relative_reward_bonus * torch.clamp(
        curriculum_progress + mid_curriculum,
        0.0,
        1.0,
    )
    grasp_object_scale = 1.0 + grasp_object_bonus * grasp_phase
    task_scale = 1.0 + late_task_reward_bonus * curriculum_progress + manipulation_task_bonus * manipulation_phase
    late_no_contact = curriculum_progress * no_contact_sustain
    late_contact_reward_gate = 1.0 - (1.0 - no_contact_late_reward_floor) * late_no_contact
    anchor_object_gate = anchor_object_gate_floor + (1.0 - anchor_object_gate_floor) * (
        (1.0 - curriculum_progress) + curriculum_progress * proximity_gate
    )

    hand_reward = torch.exp(-hand_reward_scale * hand_kpt_err)
    hand_pos_reward = torch.exp(-hand_pos_reward_scale * hand_pos_err)
    hand_anchor_reward = torch.exp(-hand_anchor_reward_scale * hand_anchor_err)
    hand_obj_offset_reward = torch.exp(-hand_obj_offset_reward_scale * hand_obj_offset_err)
    anchor_obj_offset_reward = torch.exp(-anchor_obj_offset_reward_scale * anchor_obj_offset_err)
    hand_dof_reward = torch.exp(-hand_dof_reward_scale * hand_dof_err)
    anchor_position_gate = torch.exp(-anchor_rotation_gate_scale * hand_pos_err)
    anchor_rotation_gate = torch.maximum(anchor_position_gate, proximity_gate)
    hand_rot_reward = anchor_rotation_gate * torch.exp(-hand_rot_reward_scale * hand_rot_err)
    finger_shape_reward = torch.exp(-finger_shape_reward_scale * finger_shape_err)
    finger_topology_reward = torch.exp(-finger_topology_reward_scale * finger_topology_err)
    fingertip_reward = torch.exp(-fingertip_reward_scale * fingertip_err)
    fingertip_obj_proximity_reward = torch.exp(-fingertip_obj_proximity_reward_scale * fingertip_obj_topk_proximity_err)
    fingertip_obj_offset_reward = torch.exp(-fingertip_obj_offset_reward_scale * fingertip_obj_offset_err)
    contact_force_reward = torch.clamp(contact_force, 0.0, contact_reward_max_force) / (contact_reward_max_force + 1.0e-6)
    contact_count_reward = torch.clamp(num_contact_fingers / (target_contact_fingers + 1.0e-6), 0.0, 1.0)
    contact_sustain_reward = torch.clamp(contact_sustain, 0.0, 1.0)
    thumb_ref_weight = torch.clamp(ref_fingertip_contact_weights[:, 0], 0.0, 1.0)
    thumb_contact = contact_active[:, 0]
    thumb_opposition_gate = (1.0 - thumb_ref_weight) + thumb_ref_weight * thumb_contact
    non_thumb_ref_weights = ref_fingertip_contact_weights[:, 1:]
    non_thumb_ref_target = non_thumb_ref_weights.sum(dim=-1)
    non_thumb_ref_presence = torch.clamp(non_thumb_ref_target, 0.0, 1.0)
    non_thumb_contact_score = (contact_active[:, 1:] * non_thumb_ref_weights).sum(dim=-1)
    non_thumb_contact_gate = (1.0 - non_thumb_ref_presence) + non_thumb_ref_presence * torch.clamp(
        non_thumb_contact_score / (non_thumb_ref_target + 1.0e-6),
        0.0,
        1.0,
    )
    force_dominance = max_contact_force / (contact_force + 1.0e-6)
    force_balance_reward = torch.clamp(
        (force_dominance_limit - force_dominance)
        / (force_dominance_limit - 1.0 / target_contact_fingers + 1.0e-6),
        0.0,
        1.0,
    )
    stable_grasp_gate = thumb_opposition_gate * non_thumb_contact_gate * force_balance_reward * contact_sustain_reward
    contact_reward = proximity_gate * (
        contact_force_reward_weight * contact_force_reward
        + contact_count_reward_weight * contact_count_reward
        + contact_sustain_reward_weight * contact_sustain_reward
    )
    obj_pos_reward = torch.exp(-obj_pos_reward_scale * obj_pos_err)
    obj_delta_reward = torch.exp(-obj_delta_reward_scale * obj_delta_err)
    obj_rot_reward = torch.exp(-obj_rot_reward_scale * obj_rot_err)
    obj_vel_reward = torch.exp(-obj_vel_reward_scale * obj_vel_err)
    object_gate = object_reward_gate_base + (1.0 - object_reward_gate_base) * proximity_gate
    stable_grasp_reward = stable_grasp_gate * finger_topology_reward * fingertip_obj_offset_reward
    transport_gate = 0.20 * contact_sustain_reward + 0.80 * stable_grasp_gate
    transport_support_reward = transport_gate * finger_topology_reward * (
        0.35 * obj_pos_reward
        + 0.30 * obj_delta_reward
        + 0.25 * obj_rot_reward
        + 0.10 * obj_future_dir_reward
    )
    frame0_approach_gate = (start_frame_idx == 0).float() * approach_phase
    finger_shape_phase_scale = approach_phase + grasp_phase + finger_shape_contact_decay * manipulation_phase
    finger_topology_phase_scale = approach_phase + grasp_phase + finger_topology_contact_decay * manipulation_phase
    pre_contact_pose_gate = torch.clamp(1.0 - contact_sustain_reward, 0.0, 1.0)
    pre_contact_pose_bonus = pre_contact_pose_gate * (
        0.30 * hand_pos_reward
        + 0.25 * hand_anchor_reward
        + 0.20 * hand_rot_reward
        + 0.25 * finger_shape_reward
    )
    object_relative_quality_gate = (1.0 - manipulation_phase) + manipulation_phase * (
        0.25 * contact_sustain_reward
        + 0.75 * stable_grasp_gate * finger_topology_reward * obj_rot_reward
    )
    grasped_hand_ref_reward = manipulation_phase * contact_sustain_reward * finger_topology_reward * (
        0.35 * hand_pos_reward
        + 0.25 * hand_anchor_reward
        + 0.25 * hand_rot_reward
        + 0.15 * hand_dof_reward
    )
    successful_grasp_shape_reward = (
        0.5 * hand_dof_reward
        + 0.3 * fingertip_reward
        + 0.2 * fingertip_obj_offset_reward
    )
    successful_grasp_shape_bonus = stable_grasp_gate * finger_topology_reward * fingertip_obj_offset_reward * successful_grasp_shape_reward

    action_penalty = torch.sum(actions * actions, dim=-1)
    no_grasp_rotation_penalty = (
        curriculum_progress
        * no_contact_sustain
        * (1.0 - proximity_gate)
        * torch.norm(actions[:, 3:9], p=2, dim=-1)
    )
    early_lag_penalty = early_episode_gate * (hand_pos_err + hand_anchor_err)

    reward = (
        no_contact_mano_gate * position_imitation_scale * mano_regrasp_scale * hand_pos_weight * hand_pos_reward
        + no_contact_mano_gate * pose_imitation_scale * mano_regrasp_scale * hand_anchor_weight * hand_anchor_reward
        + object_relative_quality_gate * grasp_object_scale * object_relative_scale * hand_obj_offset_weight * hand_obj_offset_reward
        + object_relative_quality_gate * grasp_object_scale * object_relative_scale * anchor_obj_offset_weight * anchor_obj_offset_reward
        + no_contact_mano_gate * pose_imitation_scale * hand_dof_weight * hand_dof_reward
        + no_contact_mano_gate * late_contact_reward_gate * pose_imitation_scale * hand_weight * hand_reward
        + no_contact_mano_gate * late_contact_reward_gate * mano_regrasp_scale * anchor_object_gate * hand_rot_weight * hand_rot_reward
        + no_contact_mano_gate * pose_imitation_scale * finger_shape_phase_scale * finger_shape_weight * finger_shape_reward
        + no_contact_mano_gate * pose_imitation_scale * finger_topology_phase_scale * finger_topology_weight * finger_topology_reward
        + no_contact_mano_gate * late_contact_reward_gate * pose_imitation_scale * fingertip_weight * fingertip_reward
        + grasp_object_scale * fingertip_obj_proximity_weight * fingertip_obj_proximity_reward
        + object_relative_quality_gate * grasp_object_scale * object_relative_scale * fingertip_obj_offset_weight * fingertip_obj_offset_reward
        + grasp_object_scale * task_scale * contact_weight * contact_reward
        + grasp_object_scale * task_scale * stable_grasp_reward_weight * stable_grasp_reward
        + object_gate * obj_pos_weight * obj_pos_reward
        + object_gate * obj_rot_weight * obj_rot_reward
        + object_gate * obj_vel_weight * obj_vel_reward
        + task_scale * transport_support_reward_weight * transport_support_reward
        + task_scale * obj_future_dir_reward_weight * stable_grasp_gate * obj_future_dir_reward
        + task_scale * grasped_hand_ref_reward_weight * grasped_hand_ref_reward
        + pre_contact_pose_bonus_weight * pre_contact_pose_bonus
        + successful_grasp_dof_bonus_weight * successful_grasp_shape_bonus
        + action_penalty_scale * action_penalty
        - no_grasp_rotation_penalty_weight * no_grasp_rotation_penalty
        - early_lag_penalty_weight * early_lag_penalty
    )

    reward = torch.clamp_min(reward, 0.0)

    logs_dict = {
        "reward/total": reward,
        "reward/hand": hand_reward,
        "reward/hand_pos": hand_pos_reward,
        "reward/hand_anchor": hand_anchor_reward,
        "reward/hand_obj_offset": hand_obj_offset_reward,
        "reward/anchor_obj_offset": anchor_obj_offset_reward,
        "reward/hand_dof": hand_dof_reward,
        "reward/hand_rot": hand_rot_reward,
        "reward/finger_shape": finger_shape_reward,
        "reward/finger_topology": finger_topology_reward,
        "reward/fingertip": fingertip_reward,
        "reward/fingertip_obj_proximity": fingertip_obj_proximity_reward,
        "reward/fingertip_obj_offset": fingertip_obj_offset_reward,
        "reward/contact": contact_reward,
        "reward/contact_force": contact_force_reward,
        "reward/contact_count": contact_count_reward,
        "reward/contact_sustain": contact_sustain_reward,
        "reward/stable_grasp": stable_grasp_reward,
        "reward/transport_support": transport_support_reward,
        "reward/object_future_dir": obj_future_dir_reward,
        "reward/pre_contact_pose_bonus": pre_contact_pose_bonus,
        "reward/successful_grasp_shape_bonus": successful_grasp_shape_bonus,
        "reward/object_pos": obj_pos_reward,
        "reward/object_delta": obj_delta_reward,
        "reward/object_rot": obj_rot_reward,
        "reward/object_vel": obj_vel_reward,
        "reward/grasped_hand_ref": grasped_hand_ref_reward,
        "metric/proximity_gate": proximity_gate,
        "metric/object_gate": object_gate,
        "metric/curriculum_progress": curriculum_progress,
        "metric/mid_curriculum": mid_curriculum,
        "metric/approach_phase": approach_phase,
        "metric/frame0_approach_gate": frame0_approach_gate,
        "metric/pre_contact_pose_gate": pre_contact_pose_gate,
        "metric/grasp_phase": grasp_phase,
        "metric/manipulation_phase": manipulation_phase,
        "metric/no_contact_mano_gate": no_contact_mano_gate,
        "metric/finger_shape_phase_scale": finger_shape_phase_scale,
        "metric/finger_topology_phase_scale": finger_topology_phase_scale,
        "metric/position_imitation_scale": position_imitation_scale,
        "metric/pose_imitation_scale": pose_imitation_scale,
        "metric/mano_regrasp_scale": mano_regrasp_scale,
        "metric/grasp_object_scale": grasp_object_scale,
        "metric/object_relative_scale": object_relative_scale,
        "metric/task_scale": task_scale,
        "metric/anchor_position_gate": anchor_position_gate,
        "metric/anchor_rotation_gate": anchor_rotation_gate,
        "metric/anchor_object_gate": anchor_object_gate,
        "metric/late_contact_reward_gate": late_contact_reward_gate,
        "metric/stable_grasp_gate": stable_grasp_gate,
        "metric/transport_gate": transport_gate,
        "metric/thumb_opposition_gate": thumb_opposition_gate,
        "metric/non_thumb_contact_gate": non_thumb_contact_gate,
        "metric/force_balance": force_balance_reward,
        "metric/force_dominance": force_dominance,
        "metric/object_relative_quality_gate": object_relative_quality_gate,
        "metric/no_contact_sustain": no_contact_sustain,
        "metric/contact_fingers": num_contact_fingers,
        "metric/contact_sustain": contact_sustain_reward,
        "penalty/action": action_penalty,
        "penalty/no_grasp_rotation": no_grasp_rotation_penalty,
        "penalty/early_lag": early_lag_penalty,
        "error/hand_kpt": hand_kpt_err,
        "error/hand_pos": hand_pos_err,
        "error/hand_anchor": hand_anchor_err,
        "error/hand_obj_offset": hand_obj_offset_err,
        "error/anchor_obj_offset": anchor_obj_offset_err,
        "error/hand_dof": hand_dof_err,
        "error/hand_rot": hand_rot_err,
        "error/finger_shape": finger_shape_err,
        "error/finger_topology": finger_topology_err,
        "error/fingertip": fingertip_err,
        "error/fingertip_obj_proximity": fingertip_obj_proximity_err,
        "error/fingertip_obj_proximity_topk": fingertip_obj_topk_proximity_err,
        "error/fingertip_obj_offset": fingertip_obj_offset_err,
        "error/object_pos": obj_pos_err,
        "error/object_delta": obj_delta_err,
        "error/object_rot": obj_rot_err,
        "metric/contact_force": contact_force,
        "metric/contact_force_projected": projected_contact_force,
        "metric/contact_force_max": max_contact_force,
    }
    
    return reward, logs_dict



# Utils
def build_mano_to_shadow_dof_seq(
    mano_kpts_pos_seq: torch.Tensor,
    num_hand_dof: int,
    actuated_dof_indices: list[int],
    dof_lower_limits: torch.Tensor,
    dof_upper_limits: torch.Tensor,
) -> torch.Tensor:
    seq_len = mano_kpts_pos_seq.shape[0]
    dof_seq = torch.zeros((seq_len, num_hand_dof), device=mano_kpts_pos_seq.device)

    # MANO chains: thumb, index, middle, ring, little.
    thumb = mano_kpts_pos_seq[:, [0, 1, 2, 3, 4]]
    index = mano_kpts_pos_seq[:, [0, 5, 6, 7, 8]]
    middle = mano_kpts_pos_seq[:, [0, 9, 10, 11, 12]]
    ring = mano_kpts_pos_seq[:, [0, 13, 14, 15, 16]]
    little = mano_kpts_pos_seq[:, [0, 17, 18, 19, 20]]

    def chain_bends(chain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bend0 = joint_bend_norm(chain[:, 0], chain[:, 1], chain[:, 2])
        bend1 = joint_bend_norm(chain[:, 1], chain[:, 2], chain[:, 3])
        bend2 = joint_bend_norm(chain[:, 2], chain[:, 3], chain[:, 4])
        return bend0, bend1, bend2

    index_bends = chain_bends(index)
    middle_bends = chain_bends(middle)
    ring_bends = chain_bends(ring)
    little_bends = chain_bends(little)
    thumb_bends = chain_bends(thumb)

    zero = torch.zeros(seq_len, device=mano_kpts_pos_seq.device)
    actuated_norm = (
        zero,
        index_bends[0],
        0.5 * (index_bends[1] + index_bends[2]),
        zero,
        middle_bends[0],
        0.5 * (middle_bends[1] + middle_bends[2]),
        zero,
        ring_bends[0],
        0.5 * (ring_bends[1] + ring_bends[2]),
        zero,
        zero,
        little_bends[0],
        0.5 * (little_bends[1] + little_bends[2]),
        zero,
        thumb_bends[0],
        zero,
        thumb_bends[1],
        thumb_bends[2],
    )

    for value, dof_id in zip(actuated_norm, actuated_dof_indices):
        dof_seq[:, dof_id] = flexion_target(value, dof_lower_limits[dof_id], dof_upper_limits[dof_id])

    return dof_seq


def joint_bend_norm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    prev_bone = F.normalize(a - b, dim=-1)
    next_bone = F.normalize(c - b, dim=-1)
    cos_angle = torch.clamp((prev_bone * next_bone).sum(dim=-1), -1.0, 1.0)
    bend = torch.pi - torch.acos(cos_angle)
    return torch.clamp(bend / (0.5 * torch.pi), 0.0, 1.0)


def flexion_target(norm_value: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    lower_value = lower.item()
    upper_value = upper.item()
    if upper_value <= 0.0 and lower_value < 0.0:
        target = norm_value * lower
    elif lower_value < 0.0 < upper_value:
        target = norm_value * upper
    else:
        target = lower + norm_value * (upper - lower)
    return torch.minimum(torch.maximum(target, lower), upper)


def build_anchor_rot_ref_seq(mano_kpts_pos_seq: torch.Tensor, hand_rot_reset: torch.Tensor) -> torch.Tensor:
    wrist = mano_kpts_pos_seq[:, 0]
    index_base = mano_kpts_pos_seq[:, 5]
    middle_base = mano_kpts_pos_seq[:, 9]
    little_base = mano_kpts_pos_seq[:, 17]

    x_axis = F.normalize(index_base - little_base, dim=-1)
    y_hint = F.normalize(middle_base - wrist, dim=-1)
    z_axis = F.normalize(torch.cross(x_axis, y_hint, dim=-1), dim=-1)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)

    mano_rot_seq = quat_from_matrix(torch.stack((x_axis, y_axis, z_axis), dim=-2))
    mano_delta_seq = quat_mul(
        mano_rot_seq,
        quat_conjugate(mano_rot_seq[0:1]).expand_as(mano_rot_seq),
    )
    hand_rot_ref_seq = quat_mul(
        mano_delta_seq,
        hand_rot_reset.unsqueeze(0).expand_as(mano_delta_seq),
    )
    return F.normalize(hand_rot_ref_seq, dim=-1)


def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(matrix_from_quat(F.normalize(quat, dim=-1)))


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return axis_angle_from_quat(quat_from_matrix(matrix))
