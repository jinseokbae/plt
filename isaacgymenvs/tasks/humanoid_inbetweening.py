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

from isaacgymenvs.tasks.humanoid_imitate import HumanoidImitate, compute_imitation_reset_mean
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
)

# Obses
# NUM_OBS - Proprioceptive states (defined in humanoid_amp_base.py)
NUM_KEYFRAME_OBS = 15 * (3 + 6) * 2  # version 2 - global coordinate features : relative to sim
MINIMUM_KEYFRAME_INTERVAL = 15 # in frames (1s)
assert MINIMUM_KEYFRAME_INTERVAL % 2 != 0 # for safe noise addition

DISPLAY_REFERENCE = False # White Agent
class HumanoidInbetweening(HumanoidImitate):
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
        self.num_keyframes = cfg["env"].get("numKeyframes", None)

        # keyframe related configs
        self._kf_temporal_style = cfg["env"]["keyframeTemporalStyle"]
        if not cfg["env"]["test"]:
            assert self._kf_temporal_style in ["uniform", "chunk_random", "full_random", "full_random_long"]
        else:
            if "full_random" in self._kf_temporal_style:
                self._kf_temporal_style = "full_random_long"
            cfg["env"]["enableEarlyTermination"] = False
        self._kf_spatial_style = cfg["env"]["keyframeSpatialStyle"]
        assert self._kf_spatial_style in ["full", "masked"]
        _kf_spatial_mask_ratio = cfg["env"]["keyframeSpatialMaskRatio"]
        assert _kf_spatial_mask_ratio >= 0 and _kf_spatial_mask_ratio <= 1
        self._add_kf_interval_noise = cfg["env"]["addKeyframeIntervalNoise"]

        # only using subset
        if cfg["env"]["motion_file"] in ["LaFAN1", "AMASS"]:
            # exclude keys
            self.exclude_motion_keys = cfg["env"].get("exclude_motion_key", ["obstacles", "ground", "push"])
        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        # Keyframe settings
        # setup for keyframe buffer
        self._avg_keyframe_interval_in_frames = (self.max_episode_length // (self.num_keyframes - 1))
        self._keyframe_indices = torch.LongTensor(
            [k * self._avg_keyframe_interval_in_frames for k in range(self.num_keyframes)],
        ).to(self.device)
        self._keyframe_indices = self._keyframe_indices[None].repeat(self.num_envs, 1) # internal knowledge of agent on keyframe status
        self._uniform_keyframe_indices = self._keyframe_indices.clone()
        self._kf_change_buf = torch.zeros_like(self.reset_buf)
        self._kf_root_pos_buf = self._ref_root_states_buf[self.all_env_ids[:, None], self._keyframe_indices, :3]
        self._kf_root_rot_buf = self._ref_root_states_buf[self.all_env_ids[:, None], self._keyframe_indices, 3:7]
        self._kf_dof_pos_buf = self._ref_dof_pos_buf[self.all_env_ids[:, None], self._keyframe_indices]
        self._kf_key_pos_buf = self._ref_key_pos_buf[self.all_env_ids[:, None], self._keyframe_indices]
        self._kf_rigid_body_pos_buf = self._ref_rigid_body_pos_buf[
            self.all_env_ids[:, None], self._keyframe_indices
        ]
        self._kf_rigid_body_rot_buf = self._ref_rigid_body_rot_buf[
            self.all_env_ids[:, None], self._keyframe_indices
        ]
        self._kf_rigid_body_vel_buf = self._ref_rigid_body_vel_buf[
            self.all_env_ids[:, None], self._keyframe_indices
        ]
        self._kf_rigid_body_ang_vel_buf = self._ref_rigid_body_ang_vel_buf[
            self.all_env_ids[:, None], self._keyframe_indices
        ]
        self._curr_keyframe_indices = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._kf_spatial_survival_mask = torch.ones(self.num_envs, self._keyframe_indices.shape[1], self.humanoid_num_bodies, 1, dtype=torch.float32, device=self.device) 
        if self._kf_spatial_style == "masked":
            self._kf_spatial_survival_probs = torch.ones_like(self._kf_spatial_survival_mask) * (1 - _kf_spatial_mask_ratio)
        self._episodic_reward_rescale = self.cfg["env"]["episodicRewardRescale"]
        if self._episodic_reward_rescale:
            self._frame_ratio = torch.zeros_like(self.rew_buf)

        # positional encoding - reference from https://github.dev/jihoonerd/Robust-Motion-In-betweening
        self._positional_encoding_style = self.cfg["env"]["PositionalEncoding"]
        assert self._positional_encoding_style in ["none", "time", "ground_dist"]
        if self._positional_encoding_style == "time":
            self._time_to_arrival = (
                torch.ones(self.num_envs, device=self.device, dtype=torch.long)
                * self._avg_keyframe_interval_in_frames
            )
            max_len = self.max_episode_length + 1
            dimension = NUM_KEYFRAME_OBS  # for keyframe obs
            self._positional_encoding = torch.zeros(max_len, dimension, device=self.device, dtype=torch.float)
            position = torch.arange(0, max_len, step=1, device=self.device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dimension, 2, device=self.device).float() * (-np.log(10000.0) / dimension)
            )
            self._positional_encoding[:, 0::2] = torch.sin(position * div_term)
            if dimension % 2 == 0:
                self._positional_encoding[:, 1::2] = torch.cos(position * div_term)
            else:
                self._positional_encoding[:, 1::2] = torch.cos(position * div_term[:-1])

            # 3.3 This means that when dealing with transitions of length T max (trans), the model sees a constant
            # z tta for 5 frames before it starts to vary.
            assert max_len >= 5
            ztta_const_part = self._positional_encoding[max_len - 5]
            self._positional_encoding[max_len - 4] = ztta_const_part
            self._positional_encoding[max_len - 3] = ztta_const_part
            self._positional_encoding[max_len - 2] = ztta_const_part
            self._positional_encoding[max_len - 1] = ztta_const_part
        return

    def _load_motion(self, motion_file):
        super()._load_motion(motion_file)
        if self.cfg["env"]["test"]: # to faithfully eval
            self.env_rng_cpu = torch.Generator()
            self.env_rng_cpu.manual_seed(self.cfg["env"]["seed"])
        else:
            self.env_rng_cpu = None
        return

    def _prepare_else_assets(self):
        # NOTE THAT HUMANOID has priority
        # Ex) MUST DEFINE HUMANOID HANDLE AND THAN OBJECTS
        else_assets, else_start_poses, else_num_bodies, else_num_shapes = super()._prepare_else_assets()
        if self._display_reference and not self.cfg["env"]["kinematic"]:
            # assets for keyframes
            humanoid_asset_options = gymapi.AssetOptions()
            humanoid_asset_file = self.humanoid_asset_file
            humanoid_asset_options.fix_base_link = True
            humanoid_asset_options.disable_gravity = True
            keyframe_asset = self.gym.load_asset(
                self.sim, self.humanoid_asset_root, humanoid_asset_file, humanoid_asset_options
            )
            num_keyframe_bodies = self.gym.get_asset_rigid_body_count(keyframe_asset)
            num_keyframe_shapes = self.gym.get_asset_rigid_shape_count(keyframe_asset)
            else_assets = else_assets + [keyframe_asset for _ in range(self.num_keyframes)]
            else_start_poses = else_start_poses + [self.humanoid_start_pose for _ in range(self.num_keyframes)]
            else_num_bodies = else_num_bodies + [num_keyframe_bodies for _ in range(self.num_keyframes)]
            else_num_shapes = else_num_shapes + [num_keyframe_shapes for _ in range(self.num_keyframes)]
            self.num_else_actor += self.num_keyframes
        return else_assets, else_start_poses, else_num_bodies, else_num_shapes

    def _create_else_actors(self, env_ptr, env_idx, else_assets, else_start_poses):
        else_handles = super()._create_else_actors(env_ptr, env_idx, else_assets, else_start_poses)
        if self._display_reference and not self.cfg["env"]["kinematic"]:
            offset = 1 if hasattr(self, "ground_asset") else 0
            contact_filter = (
                1  # >1 : ignore all the collision, 0 : enable all collision, -1 : collision defined by robot file
            )
            # handle for keyframe agent
            for k in range(1, self.num_keyframes + 1):
                segmentation_id = k + 1 # segmentation ID used in segmentation camera sensors
                handle = self.gym.create_actor(
                    env_ptr,
                    else_assets[k + offset],
                    else_start_poses[k + offset],
                    "keyframe_%d" % k,
                    self.num_envs * (k + 1) + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                rate = (k - 1) / self.num_keyframes
                humanoid_color = gymapi.Vec3(0.9 *(1 - rate), 0.9 * rate, rate * 0.3)
                for j in range(self.humanoid_num_bodies):
                    self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL, humanoid_color)
                else_handles.append(handle)

        return else_handles

    def post_physics_step(self):
        self.progress_buf += 1
        if self._positional_encoding_style == "time":
            self._time_to_arrival = torch.maximum(self._time_to_arrival - 1, torch.zeros_like(self._time_to_arrival))

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        # eval metric
        if self.eval_metric:
            self.eval_tracking_metric()
        return

    def eval_tracking_metric(self):
        super().eval_tracking_metric()
        # only measuring keyframe indices
        if self.progress_buf[0] == self.max_episode_length - 1:
            mpjpe = self.metric["mpjpe_l"].reshape(self.num_envs, self.max_episode_length, -1)
            gmpjpe = self.metric["mpjpe_g"].reshape(self.num_envs, self.max_episode_length, -1)
            vel_dist = self.metric["vel_dist"].reshape(self.num_envs, self.max_episode_length - 1)
            accel_dist = self.metric["accel_dist"].reshape(self.num_envs, self.max_episode_length - 2)

            keyframe_indices = self._keyframe_indices
            mpjpe = torch.from_numpy(mpjpe).to(self.device)[self.all_env_ids[:, None], keyframe_indices]
            gmpjpe = torch.from_numpy(gmpjpe).to(self.device)[self.all_env_ids[:, None], keyframe_indices]
            vel_dist = torch.from_numpy(vel_dist).to(self.device)[self.all_env_ids[:, None], keyframe_indices - 1]
            accel_dist = torch.from_numpy(accel_dist).to(self.device)[self.all_env_ids[:, None], keyframe_indices - 2]

            mpjpe = mpjpe.view(-1, self.humanoid_num_bodies).detach().cpu().numpy()
            gmpjpe = gmpjpe.view(-1, self.humanoid_num_bodies).detach().cpu().numpy()
            vel_dist = vel_dist.view(-1).detach().cpu().numpy()
            accel_dist = accel_dist.view(-1).detach().cpu().numpy()
            
            self.metric["mpjpe_l"] = mpjpe
            self.metric["mpjpe_g"] = gmpjpe
            self.metric["vel_dist"] = vel_dist
            self.metric["accel_dist"] = accel_dist

        return
    
    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        if self._episodic_reward_rescale:
            self.rew_buf[:] = self.rew_buf * self._frame_ratio


    def _compute_keyframe_observations(self, env_ids, indices):
        if env_ids is None:
            kf_rigid_body_pos = self._kf_rigid_body_pos_buf[self.all_env_ids, indices]
            kf_rigid_body_rot = self._kf_rigid_body_rot_buf[self.all_env_ids, indices]
            sim_rigid_body_pos = self._humanoid_rigid_body_pos
            sim_rigid_body_rot = self._humanoid_rigid_body_rot
            sim_root_rot = self._humanoid_root_states[:, 3:7]
            spatial_survival_mask = self._kf_spatial_survival_mask[self.all_env_ids, indices]
        else:
            kf_rigid_body_pos = self._kf_rigid_body_pos_buf[env_ids, indices[env_ids]]
            kf_rigid_body_rot = self._kf_rigid_body_rot_buf[env_ids, indices[env_ids]]
            sim_rigid_body_pos = self._humanoid_rigid_body_pos[env_ids]
            sim_rigid_body_rot = self._humanoid_rigid_body_rot[env_ids]
            sim_root_rot = self._humanoid_root_states[env_ids, 3:7]
            spatial_survival_mask = self._kf_spatial_survival_mask[env_ids, indices[env_ids]]

        kf_obs = compute_keyframe_global_relative_observations(
            kf_rigid_body_pos,
            kf_rigid_body_rot,
            sim_rigid_body_pos,
            sim_rigid_body_rot,
            sim_root_rot,
            spatial_survival_mask
        )
        return kf_obs

    def _compute_goal_observations(self, env_ids=None):
        curr_kf_obs = self._compute_keyframe_observations(env_ids, self._curr_keyframe_indices)
        next_kf_obs = self._compute_keyframe_observations(env_ids, self._curr_keyframe_indices + 1)
        goal_obs = torch.cat([curr_kf_obs, next_kf_obs], dim=-1)
        if self._positional_encoding_style == "time":
            if env_ids is None:
                pe = self._positional_encoding[self._time_to_arrival]
            else:
                pe = self._positional_encoding[self._time_to_arrival[env_ids]]
            goal_obs = goal_obs + pe

        return goal_obs

    @property
    def num_goal_obs(self) -> int:
        return NUM_KEYFRAME_OBS

    def _randomize_keyframes(self, env_ids, mode="uniform", min_kf_interval=MINIMUM_KEYFRAME_INTERVAL):
        if mode == "uniform":
            keyframe_indices = self._uniform_keyframe_indices[env_ids]
        elif mode == "uniform_vel":
            kf_speed = (self._kf_rigid_body_pos_buf[env_ids, 1:] - self._kf_rigid_body_pos_buf[env_ids, :-1]).pow(2).sum(dim=-1).sqrt().mean(dim=-1) # (B, N - 1)
            kf_delta = kf_speed / (kf_speed.sum(dim=-1, keepdim=True))
            kf_delta = kf_delta * self.max_episode_length
            keyframe_indices = torch.cumsum(kf_delta, dim=-1).round().long()
            keyframe_indices = torch.cat([torch.zeros_like(keyframe_indices[:, :1]), keyframe_indices], dim=-1)
        elif mode == "full_random":               
            B, T, N, M = env_ids.shape[0], self.max_episode_length, self.num_keyframes, min_kf_interval
            keyframe_indices = torch.sort(torch.multinomial(torch.arange(T - M * (N - 1), device=self.device)[None].repeat(B, 1).float(), N), dim=-1).values
            keyframe_indices = keyframe_indices + torch.arange(N, device=self.device)[None].repeat(B, 1) * M
        # elif mode == "full_random_long":               
        #     B, T, N, M = env_ids.shape[0], self.max_episode_length, self.num_keyframes, self.max_episode_length // self.num_keyframes
        #     keyframe_indices = torch.sort(torch.multinomial(torch.arange(T - M * (N - 1), device=self.device)[None].repeat(B, 1).float(), N), dim=-1).values
        #     keyframe_indices = keyframe_indices + torch.arange(N, device=self.device)[None].repeat(B, 1) * M
        elif mode == "full_random_long":               
            B, T, N, M = env_ids.shape[0], self.max_episode_length, self.num_keyframes, min_kf_interval
            free_space = T - 1 - M * (N - 1)
            random_points = torch.sort(torch.randint(0, free_space + 1, (B, N - 2, ), generator=self.env_rng_cpu)).values.to(self.device) # (B, N - 2)
            random_points = random_points + torch.arange(1, N - 1, device=self.device)[None].repeat(B, 1) * M
            keyframe_indices = torch.cat([torch.zeros(B, 1, dtype=torch.int64, device=self.device), random_points, torch.ones(B, 1, dtype=torch.int64, device=self.device) * (T - 1)], dim=-1)
        elif mode == "full_random_two":
            B, T, _, M = env_ids.shape[0], self.max_episode_length, self.num_keyframes, self.max_episode_length // 2
            keyframe_indices = torch.zeros(B, 2, dtype=torch.int64, device=self.device)
            rand_int = torch.randint(low=MINIMUM_KEYFRAME_INTERVAL, high=self.max_episode_length, size=(1,), device=self.device, generator=self.env_rng)
            keyframe_indices[:, 1] = rand_int
        return keyframe_indices

    def _register_keyframe_indices(self, env_ids, keyframe_indices, min_kf_interval=MINIMUM_KEYFRAME_INTERVAL):
        # to set self._keyframe_indices : which is the knowledge about the keyframe inidces for the agent
        # set keyframe obs
        if self._add_kf_interval_noise:
            # generate noise
            noise = torch.randint(* keyframe_indices[:, 1:].shape, high=min_kf_interval, generator=self.env_rng) - min_kf_interval // 2
            noise[:, -1] =  - noise[:, -1].abs() # ensure not to over max_episode_length for the last keyframe
            # add noise to original keyframe intervals
            keyframe_intervals = keyframe_indices[:, 1:] - keyframe_indices[:, :-1]
            keyframe_intervals = keyframe_intervals + noise
            # reconstruct the indices
            new_keyframe_indices = torch.cumsum(keyframe_intervals, dim=-1)
            new_keyframe_indices = torch.cat([torch.zeros_like(new_keyframe_indices[:, :1]), new_keyframe_indices], dim=-1)
            keyframe_indices = new_keyframe_indices + keyframe_indices[:, :1]

        self._keyframe_indices[env_ids] = keyframe_indices
        self._curr_keyframe_indices[env_ids] = 0
        if self._episodic_reward_rescale:
            total_interval = keyframe_indices[:, -1] - keyframe_indices[:, 0]
            assert not torch.any(total_interval <= 0)
            self._frame_ratio[env_ids] = self.max_episode_length / (total_interval + 1)

        if self._positional_encoding_style == "time":
            self._time_to_arrival[env_ids] = self._keyframe_indices[env_ids, 1] - self._keyframe_indices[env_ids, 0]
        return

    def _sample_motion_ids_and_times(self, env_ids):
        motion_ids, motion_times = super()._sample_motion_ids_and_times(env_ids)
        if self.eval_jitter:
            motion_ids[:] = 0
            motion_times[:] = 0
        return motion_ids, motion_times

    def _set_env_state(
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
        # basic setup for the real keyframe indices
        original_keyframe_indices = self._randomize_keyframes(env_ids, self._kf_temporal_style)
        all_ids = torch.arange(env_ids.shape[0], device=self.device)
        start_idx = original_keyframe_indices[:, 0]
        self._set_humanoid_state(
            env_ids=env_ids,
            root_pos=root_pos[all_ids, start_idx],
            root_rot=root_rot[all_ids, start_idx],
            dof_pos=dof_pos[all_ids, start_idx],
            root_vel=root_vel[all_ids, start_idx],
            root_ang_vel=root_ang_vel[all_ids, start_idx],
            dof_vel=dof_vel[all_ids, start_idx],
            rigid_body_pos=global_pos[all_ids, start_idx],
            rigid_body_rot=global_rot[all_ids, start_idx],
            rigid_body_vel=global_vel[all_ids, start_idx],
            rigid_body_ang_vel=global_ang_vel[all_ids, start_idx],
        )
        self._set_else_states(
            env_ids=env_ids,
            keyframe_indices=original_keyframe_indices,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_pos=global_pos,
            global_rot=global_rot,
            global_vel=global_vel,
            global_ang_vel=global_ang_vel,
        )

        # register keyframes with noise - knowledge for the agent
        self._register_keyframe_indices(env_ids, original_keyframe_indices)

        global_actor_indices = self.global_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_states),
            gymtorch.unwrap_tensor(global_actor_indices),
            len(global_actor_indices),
        )

        global_dof_actor_indices = self.global_dof_actor_indices[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(global_dof_actor_indices),
            len(global_dof_actor_indices),
        )
        return

    def _set_else_states(
        self,
        env_ids,
        keyframe_indices,
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
        # set reward buf
        self._ref_root_states_buf[env_ids] = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        self._ref_dof_pos_buf[env_ids] = dof_pos
        self._ref_dof_vel_buf[env_ids] = dof_vel
        self._ref_key_pos_buf[env_ids] = key_pos
        self._ref_rigid_body_pos_buf[env_ids] = global_pos
        self._ref_rigid_body_rot_buf[env_ids] = global_rot
        self._ref_rigid_body_vel_buf[env_ids] = global_vel
        self._ref_rigid_body_ang_vel_buf[env_ids] = global_ang_vel


        self._kf_root_pos_buf[env_ids] = self._ref_root_states_buf[env_ids[:, None], keyframe_indices, :3]
        self._kf_root_rot_buf[env_ids] = self._ref_root_states_buf[env_ids[:, None], keyframe_indices, 3:7]
        self._kf_dof_pos_buf[env_ids] = self._ref_dof_pos_buf[env_ids[:, None], keyframe_indices]
        self._kf_key_pos_buf[env_ids] = self._ref_key_pos_buf[env_ids[:, None], keyframe_indices]
        self._kf_rigid_body_pos_buf[env_ids] = self._ref_rigid_body_pos_buf[env_ids[:, None], keyframe_indices]
        self._kf_rigid_body_rot_buf[env_ids] = self._ref_rigid_body_rot_buf[env_ids[:, None], keyframe_indices]

        if self._kf_spatial_style == "masked":
            self._kf_spatial_survival_mask[env_ids] = torch.distributions.Bernoulli(probs=self._kf_spatial_survival_probs[env_ids]).sample()

        # for rendering
        if self._display_reference and not self.cfg["env"]["kinematic"]:  # only for rendering
            # first set reference motion agent
            as_idx = self._ref_actor_idx
            ae_idx = as_idx + 1
            rs_idx = self._ref_actor_idx
            re_idx = rs_idx + self.humanoid_num_bodies
            ds_idx = self._ref_dof_start_idx
            de_idx = ds_idx + self.humanoid_num_dof

            # set poses
            self._else_root_states[env_ids, as_idx: ae_idx, 0:3] = root_pos[:, :1]
            self._else_root_states[env_ids, as_idx: ae_idx, 3:7] = root_rot[:, :1]
            self._else_rigid_body_pos[env_ids, rs_idx: re_idx] = global_pos[:, 0]
            self._else_rigid_body_rot[env_ids, rs_idx: re_idx] = global_rot[:, 0]
            self._else_dof_pos[
                env_ids, ds_idx: de_idx
            ] = dof_pos[:, 0]
            # zero velocities
            self._else_root_states[env_ids, as_idx: ae_idx, 7:10] = root_vel[:, :1]
            self._else_root_states[env_ids, as_idx: ae_idx, 10:13] = root_ang_vel[:, :1]
            self._else_rigid_body_vel[env_ids, rs_idx: re_idx] = global_vel[:, 0]
            self._else_rigid_body_ang_vel[env_ids, rs_idx: re_idx] = global_ang_vel[:, 0]
            self._else_dof_vel[
                env_ids, ds_idx: de_idx
            ] = dof_vel[:, 0]
            if self._pd_control:
                self._target_actions[
                    env_ids, ds_idx: de_idx
                ] = self._else_dof_pos[
                    env_ids, ds_idx: de_idx
                ]
            if not DISPLAY_REFERENCE:
                self._else_root_states[env_ids, as_idx: ae_idx, 2] = 10000 

            # next keyframe visualization
            # keyframe agents
            as_idx = ae_idx
            ae_idx = as_idx + self.num_keyframes
            rs_idx = re_idx
            re_idx = rs_idx + self.num_keyframes * self.humanoid_num_bodies
            ds_idx = de_idx
            de_idx = ds_idx + self.num_keyframes * self.humanoid_num_dof

            # set poses
            self._else_root_states[
                env_ids, as_idx :ae_idx, 0:3
            ] = self._kf_root_pos_buf[env_ids]
            self._else_root_states[
                env_ids, as_idx :ae_idx, 3:7
            ] = self._kf_root_rot_buf[env_ids]
            self._else_rigid_body_pos[
                env_ids, rs_idx: re_idx
            ] = self._kf_rigid_body_pos_buf[env_ids].view(env_ids.shape[0], self.num_keyframes * self.humanoid_num_bodies, 3)
            self._else_rigid_body_rot[
                env_ids, rs_idx: re_idx
            ] = self._kf_rigid_body_rot_buf[env_ids].view(env_ids.shape[0], self.num_keyframes * self.humanoid_num_bodies, 4)
            self._else_dof_pos[
                env_ids,
                ds_idx: de_idx,
            ] = self._kf_dof_pos_buf[env_ids].view(env_ids.shape[0], -1)
            # zero velocities
            self._else_root_states[
                env_ids, as_idx : ae_idx, 7:10
            ] = 0
            self._else_root_states[
                env_ids, as_idx : ae_idx, 10:13
            ] = 0
            self._else_rigid_body_vel[
                env_ids, rs_idx: re_idx
            ] = self._kf_rigid_body_vel_buf[env_ids].view(env_ids.shape[0], self.num_keyframes * self.humanoid_num_bodies, 3)
            self._else_rigid_body_ang_vel[
                env_ids, rs_idx: re_idx
            ] = self._kf_rigid_body_ang_vel_buf[env_ids].view(env_ids.shape[0], self.num_keyframes * self.humanoid_num_bodies, 3)
            self._else_dof_vel[
                env_ids,
                ds_idx: de_idx,
            ] = 0
            if self._pd_control:
                self._target_actions[
                    env_ids,
                    ds_idx: de_idx
                ] = self._else_dof_pos[
                    env_ids,
                    ds_idx: de_idx
                ]
            if self.cfg["env"]["test"] and self.cfg["env"].get("white_mode", False):
                self._else_root_states[env_ids, 2, 0] = -1000 # eliminate first keyframe
                if not hasattr(self, "epi_count"):
                    self.epi_count = 0
                print(self.epi_count)
                self.epi_count += 1
        return

    def _update_curr_keyframe(self):  # helper function to update curr_keyframe_obs
        # buffer for indicating keyframe indices change added
        self._kf_change_buf[:] = 0
        change_ids = torch.where(
            self.progress_buf - (self._keyframe_indices[self.all_env_ids, self._curr_keyframe_indices + 1]) == 0
        )[0]
        if change_ids.shape[0] > 0:
            # update
            self._curr_keyframe_indices[change_ids] += 1
            self._kf_change_buf[change_ids] = 1

            if not self.headless and self.cfg["env"]["test"] and self.cfg["env"].get("white_mode", False):
                self._else_root_states[change_ids, 2 + self._curr_keyframe_indices[change_ids], 0] = -1000
                if not self.reset_call_flag:
                    remove_actor_indices = self.global_actor_indices[change_ids, 3 + self._curr_keyframe_indices[change_ids]].flatten().clone()
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self._root_states),
                        gymtorch.unwrap_tensor(remove_actor_indices),
                        len(remove_actor_indices),
                    )

        return change_ids

    def _update_time_to_arrival(self, change_ids):
        if change_ids.shape[0] > 0:
            curr_indices = self._curr_keyframe_indices[change_ids]
            next_indices = self._curr_keyframe_indices[change_ids] + 1
            keyframe_indices = self._keyframe_indices[change_ids]
            valid_ids = torch.where(next_indices < self.num_keyframes)[0]
            if valid_ids.shape[0] > 0:
                curr_keyframe_indices = keyframe_indices[valid_ids, curr_indices[valid_ids]]
                next_keyframe_indices = keyframe_indices[valid_ids, next_indices[valid_ids]]
                self._time_to_arrival[valid_ids] = next_keyframe_indices - curr_keyframe_indices
        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self.progress_buf[env_ids] = self._keyframe_indices[env_ids, 0]

    def _compute_reset(self):
        super()._compute_reset()
        # update current keyframe
        update_ids = self._update_curr_keyframe()
        if self._positional_encoding_style == "time":
            self._update_time_to_arrival(update_ids)

        reset_ids = torch.where(self._curr_keyframe_indices == self.num_keyframes - 1)[0]
        if reset_ids.shape[0] > 0:
            self.reset_buf[reset_ids] = 1
            self._terminate_buf[reset_ids] = 1
        return

    def _motion_sync(self):
        if DISPLAY_REFERENCE:
            super()._motion_sync()
        else:
            return

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        # default
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.5, 3.0)

        # static
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_keyframe_global_relative_observations(
    ref_rigid_body_pos, ref_rigid_body_rot, sim_rigid_body_pos, sim_rigid_body_rot, sim_root_rot, spatial_survival_mask
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    N, J, _ = ref_rigid_body_pos.shape
    # heading rots
    sim_heading_rot = calc_heading_quat_inv(sim_root_rot)
    sim_heading_rot_expand = sim_heading_rot[:, None].repeat(1, J, 1)
    sim_heading_rot_flat = sim_heading_rot_expand.reshape(N * J, -1)

    # spatial_survival_mask
    spatial_survival_mask = spatial_survival_mask.reshape(N * J, -1)

    # rigid body pos
    ref_rigid_body_pos_flat = ref_rigid_body_pos.reshape(N * J, -1)
    sim_rigid_body_pos_flat = sim_rigid_body_pos.reshape(N * J, -1)
    ref_rigid_body_pos_obs = my_quat_rotate(sim_heading_rot_flat, ref_rigid_body_pos_flat - sim_rigid_body_pos_flat)
    ref_rigid_body_pos_obs = (ref_rigid_body_pos_obs * spatial_survival_mask).view(N, -1)

    # rigid body rot
    ref_rigid_body_rot_flat = ref_rigid_body_rot.reshape(N * J, -1)
    sim_rigid_body_rot_flat = sim_rigid_body_rot.reshape(N * J, -1)
    ref_rigid_body_rot_obs = quat_mul(
        sim_heading_rot_flat, quat_mul(ref_rigid_body_rot_flat, quat_conjugate(sim_rigid_body_rot_flat))
    )
    ref_rigid_body_rot_obs = quat_to_rotation_6d(ref_rigid_body_rot_obs)
    ref_rigid_body_rot_obs = (ref_rigid_body_rot_obs * spatial_survival_mask).view(N, -1)

    obs = torch.cat(
        (ref_rigid_body_pos_obs, ref_rigid_body_rot_obs),
        dim=-1,
    )

    return obs
