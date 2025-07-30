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

from isaacgymenvs.tasks.amp.humanoid_amp_base import dof_to_obs
from isaacgymenvs.tasks.humanoid_imitate import HumanoidImitate
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
NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
MATCHING_DISTANCE = 10
class HumanoidImitateAMP(HumanoidImitate):
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
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2

        super().__init__(
            cfg,
            rl_device,
            sim_device,
            graphics_device_id,
            headless,
            virtual_screen_capture,
            force_render,
        )
        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP

        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        self._amp_obs_buf = torch.zeros(
            (self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP),
            device=self.device,
            dtype=torch.float,
        )
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        self._amp_obs_demo_buf = None
        return

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat
        if self.cfg["env"]["motion_matching"]:
            curr_feat = self._curr_amp_obs_buf
            hist_feat = self._hist_amp_obs_buf[:, 0]

            # normalize
            curr_feat = (curr_feat - self._motion_lib.feats_mean) / self._motion_lib.feats_std
            hist_feat = (hist_feat - self._motion_lib.feats_mean) / self._motion_lib.feats_std

            # calculate distance
            curr_dist = (curr_feat[:, None] - self._motion_lib.curr_feats[None]).pow(2).mean(dim=-1)
            hist_dist = (hist_feat[:, None] - self._motion_lib.hist_feats[None]).pow(2).mean(dim=-1)
            total_dist = curr_dist + hist_dist # (B, Nclip)

            # find best matching ones
            matched_dist, matched_idx = total_dist.min(dim=-1)
            
            # collect distance (reset when episode starts)
            if not hasattr(self, "mean_matched_dist"):
                self.mean_matched_dist = 0

            valid_ids = matched_dist < MATCHING_DISTANCE
            valid_matched_dist = matched_dist[valid_ids]
            valid_matched_ids = matched_idx[valid_ids]
            self.mean_matched_dist += valid_matched_dist.mean().item()

            # collect transition ids
            if not hasattr(self, "matched_counts"):
                # last one channel is for invalid one
                self.matched_counts = torch.zeros(self._motion_lib.curr_feats.shape[0], dtype=torch.int64, device=self.device)
                self.total_samples = 0
            torch_scatter.scatter(
                src=torch.ones_like(valid_matched_ids),
                index=valid_matched_ids,
                out=self.matched_counts
            )
            self.total_samples += matched_dist.shape[0]
        return
    
    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples):
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples)

        if self._amp_obs_demo_buf is None:
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert self._amp_obs_demo_buf.shape[0] == num_samples

        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, self._num_amp_obs_steps).to(self.device)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        (root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, *_) = self._motion_lib.get_motion_state(
            motion_ids, motion_times
        )
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos, self._local_root_obs)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape)

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros(
            (num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP),
            device=self.device,
            dtype=torch.float,
        )
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(
                self._reset_ref_env_ids,
                self._reset_ref_motion_ids,
                self._reset_ref_motion_times,
            )
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1).to(self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        (root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, *_) = self._motion_lib.get_motion_state(
            motion_ids, motion_times
        )
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos, self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._humanoid_rigid_body_pos[:, self._key_body_ids, :]
        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations(
                self._humanoid_root_states,
                self._humanoid_dof_pos,
                self._humanoid_dof_vel,
                key_body_pos,
                self._local_root_obs,
            )
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(
                self._humanoid_root_states[env_ids],
                self._humanoid_dof_pos[env_ids],
                self._humanoid_dof_vel[env_ids],
                key_body_pos[env_ids],
                self._local_root_obs,
            )
        return
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def build_amp_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    root_vel = root_states[:, 7:10]
    root_ang_vel = root_states[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_rotation_6d(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos)

    obs = torch.cat(
        (
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs