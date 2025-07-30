# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE..

import os
from enum import Enum

# for motion library
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import gym
import numpy as np
import omegaconf
import torch
import torch_scatter
from gym import spaces
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.amp.humanoid_amp_base import (
    DOF_BODY_IDS,
    DOF_OFFSETS,
    HumanoidAMPBase,
    compute_humanoid_reset,
    dof_to_obs,
)
from isaacgymenvs.tasks.humanoid_imitate import HumanoidImitate, compute_ref_diff_observations, compute_phc_reward
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.utils.torch_jit_utils import (
    calc_heading_quat,
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_conjugate,
    quat_diff_rad,
    quat_mul,
    quat_to_rotation_6d,
    slerp,
    to_torch,
    normalize
)

PLANE_CONTACT_THRESHOLD = 0.05  # at the initial frame, foot must higher than this value
# Obses
# NUM_OBS - Proprioceptive states (defined in humanoid_amp_base.py)

ET_THRESHOLD = 0.1
PREPARATION_TIME = 0
ET_TIMEWINDOW = 30

class HumanoidTracking(HumanoidImitate):

    def __init__(
        self,
        cfg,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        # exclude keys
        if cfg["env"]["motion_file"] == "LaFAN1":
            self.exclude_motion_keys = cfg["env"].get("exclude_motion_key", ["obstacles", "ground", "push"])

        num_track_points = cfg["env"]["num_track_points"]
        if num_track_points == 1:
            self._vr_body_ids = [2] # head
        elif num_track_points == 3:
            tilted = cfg["env"]["tilted"]
            if tilted:
                self._vr_body_ids = [2, 5, 11] # head / right hand / right foot
            else:
                self._vr_body_ids = [2, 5, 8] # head / right hand / right foot
        elif num_track_points == 5:
            self._vr_body_ids = [2, 5, 8, 11, 14] # head / right hand / left hand / right foot / left foot
        else:
            raise ValueError("Invalid Number for Tracking Points")
        self._vr_body_ids = torch.tensor(self._vr_body_ids, dtype=torch.int64, device=rl_device)
        self._num_next_obs_track= self._vr_body_ids.shape[-1] * (3 + 6 + 3 + 3)  # right next frame
        super().__init__(
            cfg=cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        if self._display_reference:
            self.tracker_indices = self.global_actor_indices[:, 1:].flatten().clone()
        self.head_offset = torch.tensor([0.05, 0, 0.22], dtype=torch.float32, device=self.device)
        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _prepare_else_assets(self):
        # NOTE THAT HUMANOID has priority
        # Ex) MUST DEFINE HUMANOID HANDLE AND THAN OBJECTS
        else_assets, else_start_poses, else_num_bodies, else_num_shapes = [], [], [], []
        self.num_else_actor = 0
        if self._display_reference:
            sphere_asset_options = gymapi.AssetOptions()
            sphere_asset_options.disable_gravity = True
            sphere_asset = self.gym.create_sphere(self.sim, 0.05, sphere_asset_options)

            sphere_pose = gymapi.Transform()
            sphere_pose.p = gymapi.Vec3(0, 0, 10.0)

            for _ in range(self._vr_body_ids.shape[-1]):
                else_assets.append(sphere_asset)
                else_start_poses.append(self.humanoid_start_pose)
                else_num_bodies.append(self.gym.get_asset_rigid_body_count(sphere_asset))
                else_num_shapes.append(self.gym.get_asset_rigid_shape_count(sphere_asset))
                self.num_else_actor += 1

        return else_assets, else_start_poses, else_num_bodies, else_num_shapes

    def _create_else_actors(self, env_ptr, env_idx, else_assets, else_start_poses):
        else_handles = []
        if self._display_reference:
            offset = 0
            contact_filter = (
                1  # >1 : ignore all the collision, 0 : enable all collision, -1 : collision defined by robot file
            )

            # handle for tracker
            segmentation_id = 2 # segmentation ID used in segmentation camera sensors
            for i in range(self._vr_body_ids.shape[-1]):
                handle = self.gym.create_actor(
                    env_ptr,
                    else_assets[offset],
                    else_start_poses[offset],
                    f"tracker_{i}",
                    self.num_envs * (i + 1) + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                tracker_color = gymapi.Vec3(1.0, 0.3, 0.3)
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, tracker_color)
                else_handles.append(handle)

        return else_handles

    def _compute_goal_observations(self, env_ids=None):
        goal_obs = self._compute_track_diff_observations(env_ids)
        return goal_obs

    def _compute_track_diff_observations(self, env_ids=None):
        if env_ids is None:
            B = self.num_envs
            time_indices = torch.stack([self.progress_buf + i for i in range(1, self._num_next_obs_steps + 1)], dim=-1)
            ref_rigid_body_pos = self._ref_rigid_body_pos_buf[self.all_env_ids[:, None], time_indices]
            ref_rigid_body_rot = self._ref_rigid_body_rot_buf[self.all_env_ids[:, None], time_indices]
            ref_rigid_body_vel = self._ref_rigid_body_vel_buf[self.all_env_ids[:, None], time_indices]
            ref_rigid_body_ang_vel = self._ref_rigid_body_ang_vel_buf[self.all_env_ids[:, None], time_indices]

            sim_rigid_body_pos = self._humanoid_rigid_body_pos[:, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_rot = self._humanoid_rigid_body_rot[:, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_vel = self._humanoid_rigid_body_vel[:, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_ang_vel = self._humanoid_rigid_body_ang_vel[:, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_root_rot = self._humanoid_root_states[:, None, 3:7].repeat(1, self._num_next_obs_steps, 1)
        else:
            B = env_ids.shape[0]
            time_indices = torch.stack([self.progress_buf[env_ids] + i for i in range(1, self._num_next_obs_steps + 1)], dim=-1)
            ref_rigid_body_pos = self._ref_rigid_body_pos_buf[env_ids[:, None], time_indices]
            ref_rigid_body_rot = self._ref_rigid_body_rot_buf[env_ids[:, None], time_indices]
            ref_rigid_body_vel = self._ref_rigid_body_vel_buf[env_ids[:, None], time_indices]
            ref_rigid_body_ang_vel = self._ref_rigid_body_ang_vel_buf[env_ids[:, None], time_indices]

            sim_rigid_body_pos = self._humanoid_rigid_body_pos[env_ids, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_rot = self._humanoid_rigid_body_rot[env_ids, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_vel = self._humanoid_rigid_body_vel[env_ids, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_rigid_body_ang_vel = self._humanoid_rigid_body_ang_vel[env_ids, None].repeat(1, self._num_next_obs_steps, 1, 1)
            sim_root_rot = self._humanoid_root_states[env_ids, None, 3:7].repeat(1, self._num_next_obs_steps, 1)

        J = self.humanoid_num_bodies
        ref_diff_obs = compute_ref_diff_observations(
            ref_rigid_body_pos.view(-1, J, 3)[:, self._vr_body_ids],
            ref_rigid_body_rot.view(-1, J, 4)[:, self._vr_body_ids],
            ref_rigid_body_vel.view(-1, J, 3)[:, self._vr_body_ids],
            ref_rigid_body_ang_vel.view(-1, J, 3)[:, self._vr_body_ids],
            sim_rigid_body_pos.view(-1, J, 3)[:, self._vr_body_ids],
            sim_rigid_body_rot.view(-1, J, 4)[:, self._vr_body_ids],
            sim_rigid_body_vel.view(-1, J, 3)[:, self._vr_body_ids],
            sim_rigid_body_ang_vel.view(-1, J, 3)[:, self._vr_body_ids],
            sim_root_rot.view(-1, 4),
        )
        ref_diff_obs = ref_diff_obs.view(B, -1)
        return ref_diff_obs
    
    @property
    def num_goal_obs(self) -> int:
        return self._num_next_obs_track* self._num_next_obs_steps

    def _set_else_states(
        self,
        env_ids,
        root_pos,
        root_rot,
        dof_pos,
        root_vel,
        root_ang_vel,
        dof_vel,
        key_pos,
        global_pos,
        global_rot,
        global_vel,
        global_ang_vel,
    ):
        if self._add_noise:
            scale = self._noise_scale
            global_pos += torch.empty_like(global_pos).normal_(generator=self.env_rng) * scale
            global_rot += torch.empty_like(global_rot).normal_(generator=self.env_rng) * scale
            global_rot = normalize(global_rot)
            global_vel += torch.empty_like(global_vel).normal_(generator=self.env_rng) * scale
            global_ang_vel += torch.empty_like(global_ang_vel).normal_(generator=self.env_rng) * scale

         ## set reward buf
        self._ref_root_states_buf[env_ids] = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        self._ref_dof_pos_buf[env_ids] = dof_pos
        self._ref_dof_vel_buf[env_ids] = dof_vel
        self._ref_key_pos_buf[env_ids] = key_pos
        self._ref_rigid_body_pos_buf[env_ids] = global_pos
        self._ref_rigid_body_rot_buf[env_ids] = global_rot
        self._ref_rigid_body_vel_buf[env_ids] = global_vel
        self._ref_rigid_body_ang_vel_buf[env_ids] = global_ang_vel

        # for rendering
        if self._display_reference:  # only for rendering
            # set poses for tracker
            self._else_root_states[env_ids, :self._vr_body_ids.shape[0], 0:3] = global_pos[env_ids, 0][:, self._vr_body_ids]
            self._else_root_states[env_ids, :self._vr_body_ids.shape[0], 7:10] = global_vel[env_ids, 0][:, self._vr_body_ids]
            # head offset
            head_rot = self._ref_rigid_body_rot_buf[env_ids, 0, 2] # (B, 4)
            head_offset = my_quat_rotate(head_rot, self.head_offset[None].repeat(env_ids.shape[0], 1))
            self._else_root_states[env_ids, 0, :3] += head_offset
        return

    def _motion_sync(self):
        frame_idx = self.progress_buf + 1
        # # offset for motion
        global_pos = self._ref_rigid_body_pos_buf[self.all_env_ids, frame_idx]
        global_vel = self._ref_rigid_body_vel_buf[self.all_env_ids, frame_idx]

        env_ids = self.all_env_ids
        if self._display_reference:  # agent & kinematic
            self._else_root_states[env_ids, :self._vr_body_ids.shape[0], 0:3] = global_pos[:, self._vr_body_ids]
            self._else_root_states[env_ids, :self._vr_body_ids.shape[0], 7:10] = global_vel[:, self._vr_body_ids]
            # head offset
            head_rot = self._ref_rigid_body_rot_buf[self.all_env_ids, frame_idx, 2] # (B, 4)
            head_offset = my_quat_rotate(head_rot, self.head_offset[None].repeat(self.num_envs, 1))
            self._else_root_states[env_ids, 0, :3] += head_offset

            # clipping
            self._else_root_states[env_ids, :self._vr_body_ids.shape[0], 2] = torch.maximum(
                self._else_root_states[env_ids, :self._vr_body_ids.shape[0], 2], 
                torch.ones_like(self._else_root_states[env_ids, :self._vr_body_ids.shape[0], 2]) * 0.051)

        if not self.reset_call_flag:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(self.tracker_indices),
                len(self.tracker_indices),
            )

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        
        # default
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 2.0)

        # static
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def eval_tracking_metric(self):
        super().eval_tracking_metric()
        # only measuring valid ones and report success rate
        if self.progress_buf[0] == self.max_episode_length - 1:
            mpjpe = self.metric["mpjpe_l"].reshape(self.num_envs, self.max_episode_length, -1)
            gmpjpe = self.metric["mpjpe_g"].reshape(self.num_envs, self.max_episode_length, -1)
            vel_dist = self.metric["vel_dist"].reshape(self.num_envs, self.max_episode_length - 1)
            accel_dist = self.metric["accel_dist"].reshape(self.num_envs, self.max_episode_length - 2)

            vr_body_ids = self._vr_body_ids.tolist()
            track_dist = gmpjpe[..., vr_body_ids]
            is_succeed = (track_dist.mean(axis=-1) < 1000)
            valid_mask = is_succeed
            self.metric["success_rate"] = valid_mask.sum() / (self.max_episode_length * self.num_envs)

        return

    def _compute_reset(self):
        if "track_dist_" in self.imitation_reset_style:
            curr_rigid_body_pos = self._humanoid_rigid_body_pos
            goal_rigid_body_pos = self._ref_rigid_body_pos_buf[self.all_env_ids, self.progress_buf]
            threshold = float(self.imitation_reset_style.split('_')[-1])
            assert threshold > 0
            self.reset_buf[:], self._terminate_buf[:] = compute_imitation_reset_track_dist_max(
                self.reset_buf,
                self.progress_buf,
                curr_rigid_body_pos[:, self._vr_body_ids],
                goal_rigid_body_pos[:, self._vr_body_ids],
                self.max_episode_length,
                self._enable_early_termination,
                threshold
            )
        else:
            super()._compute_reset()
        return
    
    def _compute_reward(self, actions):
        if self.imitation_rew_style != "simple_track":
            super()._compute_reward(actions)
        else:
            # joint positions (global)
            curr_rigid_body_pos = self._humanoid_rigid_body_pos
            goal_rigid_body_pos = self._ref_rigid_body_pos_buf[self.all_env_ids, self.progress_buf]

            # joint rotations (global)
            curr_rigid_body_rot = self._humanoid_rigid_body_rot
            goal_rigid_body_rot = self._ref_rigid_body_rot_buf[self.all_env_ids, self.progress_buf]

            # joint velocities (global)
            curr_rigid_body_vel = self._humanoid_rigid_body_vel
            goal_rigid_body_vel = self._ref_rigid_body_vel_buf[self.all_env_ids, self.progress_buf]

            # joint angular velocities (global)
            curr_rigid_body_ang_vel = self._humanoid_rigid_body_ang_vel
            goal_rigid_body_ang_vel = self._ref_rigid_body_ang_vel_buf[self.all_env_ids, self.progress_buf]

            self.rew_buf[:] = compute_phc_reward(
                curr_rigid_body_pos[:, self._vr_body_ids],
                goal_rigid_body_pos[:, self._vr_body_ids],
                curr_rigid_body_rot[:, self._vr_body_ids],
                goal_rigid_body_rot[:, self._vr_body_ids],
                curr_rigid_body_vel[:, self._vr_body_ids],
                goal_rigid_body_vel[:, self._vr_body_ids],
                curr_rigid_body_ang_vel[:, self._vr_body_ids],
                goal_rigid_body_ang_vel[:, self._vr_body_ids],
            )
        return

@torch.jit.script
def compute_imitation_reset_track_dist_max(
    reset_buf,
    progress_buf,
    curr_rigid_body_pos,
    goal_rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    threshold
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    # rigid_body_pos -> B, K, 3
    if enable_early_termination:
        pos_dist = (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).sqrt()
        max_pos_dist = torch.max(pos_dist, dim=-1).values
        has_deviated = max_pos_dist > threshold
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
