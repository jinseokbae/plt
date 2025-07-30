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

import numpy as np
import omegaconf
import torch
from gym import spaces
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.amp.humanoid_amp_base import DOF_BODY_IDS, DOF_OFFSETS, NUM_OBS, HumanoidAMPBase, dof_to_obs
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from isaacgymenvs.utils.torch_jit_utils import (
    calc_heading_quat_inv,
    my_quat_rotate,
    quat_diff_rad,
    quat_mul,
    quat_to_tan_norm,
    to_torch,
)

NUM_AMP_OBS_PER_STEP = 13 + 52 + 28 + 12  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
NUM_KEYFRAME_OBS = 3 + 6 + 52 + 12  # [root_pos, root_rot, dof_pos, key_body_pos]


class HumanoidAMP(HumanoidAMPBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

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
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert self._num_amp_obs_steps >= 2

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # keyframe realted configs
        self._use_keyframes = self.cfg["env"]["useKeyframes"]
        self._keyframe_interval = self.cfg["env"]["keyframeInterval"]
        if self._use_keyframes:
            # set up for motion in-betweening
            fps = round(1 / (self.cfg["sim"]["dt"] + 1e-7))  # assume fps is always integer
            control_freq_inv = self.cfg["env"]["controlFrequencyInv"]
            self._keyframe_interval_in_frames = int(fps / control_freq_inv * self._keyframe_interval)
            max_episode_length_in_time = self.cfg["env"]["episodeLength"] / (
                fps / control_freq_inv
            )  # in seconds (60 / ((1/60) / 2))
            self.truncate_time = max_episode_length_in_time  # use for sampling reference motion
            self.num_keyframes = (
                int(max_episode_length_in_time / self._keyframe_interval) + 1
            )  # 2 / 1 + 1 (should include first frame)
            self._num_obs = NUM_OBS + NUM_KEYFRAME_OBS * 2  # current obs + goal obs (current keyframe obs)
            try:
                assert NUM_KEYFRAME_OBS == 3 + 6 + 52 + 12
            except:
                raise ValueError(
                    "You need to recheck indexing of compute_keyframe_observations and compute_imitate_reward"
                )
        else:
            self._num_obs = NUM_OBS

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        motion_file = cfg["env"].get("motion_file", "amp_humanoid_backflip.npy")
        motion_file_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/motions/")

        if isinstance(motion_file, str):
            motion_file = os.path.join(motion_file_root_path, motion_file)
            # Case 1 : if it is single motion file
            if motion_file.split(".")[-1] == "npy":
                motion_file_path = [motion_file]
            # Case 2 : if it is directory
            elif os.path.isdir(motion_file):
                if motion_file.split("/")[-1] in ["LaFAN1"]:
                    subset = "test" if self.cfg["env"]["test"] else "train"
                    motion_file = os.path.join(motion_file, subset)
                motion_file_path = list(Path(motion_file).rglob("*.npy"))
            else:
                raise NotImplementedError()
        elif isinstance(motion_file, omegaconf.listconfig.ListConfig):
            motion_file_path = [os.path.join(motion_file_root_path, name) for name in motion_file]
        else:
            raise NotImplementedError()

        self._load_motion(motion_file_path)

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

        if self._use_keyframes:
            # set up for policyobs
            self._keyframe_indices = torch.LongTensor(
                [k for k in range(self.max_episode_length + 1) if k % self._keyframe_interval_in_frames == 0],
            ).to(self.device)
            self._time_to_arrival = (
                torch.ones(self.num_envs, device=self.device, dtype=torch.long) * self._keyframe_interval_in_frames
            )
            # setup for reward
            self.all_env_ids = torch.arange(self.num_envs).to(self.device)
            self._ref_root_states_buf = torch.zeros(
                self.num_envs, self.max_episode_length + 1, 13, device=self.device, dtype=torch.float32
            )
            self._ref_dof_pos_buf = torch.zeros(
                self.num_envs,
                self.max_episode_length + 1,
                self.humanoid_num_dof,
                device=self.device,
                dtype=torch.float32,
            )
            self._ref_dof_vel_buf = torch.zeros(
                self.num_envs,
                self.max_episode_length + 1,
                self.humanoid_num_dof,
                device=self.device,
                dtype=torch.float32,
            )
            self._ref_key_pos_buf = torch.zeros(
                self.num_envs,
                self.max_episode_length + 1,
                self._key_body_ids.shape[0],
                3,
                device=self.device,
                dtype=torch.float32,
            )
            self._ref_rigid_body_rot_buf = torch.zeros(
                self.num_envs,
                self.max_episode_length + 1,
                self.humanoid_num_bodies,
                4,
                device=self.device,
                dtype=torch.float32,
            )
            self._ref_rigid_body_ang_vel_buf = torch.zeros(
                self.num_envs,
                self.max_episode_length + 1,
                self.humanoid_num_bodies,
                3,
                device=self.device,
                dtype=torch.float32,
            )

            # setup for keyframe
            self._kf_root_pos_buf = self._ref_root_states_buf[:, self._keyframe_indices, :3]
            self._kf_root_rot_buf = self._ref_root_states_buf[:, self._keyframe_indices, 3:7]
            self._kf_dof_pos_buf = self._ref_dof_pos_buf[:, self._keyframe_indices]
            self._kf_key_pos_buf = self._ref_key_pos_buf[:, self._keyframe_indices]
            self._kf_rigid_body_rot_buf = self._ref_rigid_body_rot_buf[:, self._keyframe_indices]
            self._curr_keyframe_indices = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

            # positional encoding - reference from https://github.dev/jihoonerd/Robust-Motion-In-betweening
            max_len = self._keyframe_interval_in_frames
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
            # self._positional_encoding = self._positional_encoding.unsqueeze(0)

            # 3.3 This means that when dealing with transitions of length T max (trans), the model sees a constant
            # z tta for 5 frames before it starts to vary.
            assert max_len >= 5
            ztta_const_part = self._positional_encoding[0][max_len - 5]
            self._positional_encoding[0][max_len - 4] = ztta_const_part
            self._positional_encoding[0][max_len - 3] = ztta_const_part
            self._positional_encoding[0][max_len - 2] = ztta_const_part
            self._positional_encoding[0][max_len - 1] = ztta_const_part
        return

    def _prepare_else_assets(self):
        else_assets, else_start_poses, else_num_bodies, else_num_shapes = [], [], [], []
        if self._use_keyframes and self.cfg["env"]["test"]:
            keyframe_asset_options = gymapi.AssetOptions()
            keyframe_asset_options.fix_base_link = True
            keyframe_asset_options.disable_gravity = True
            keyframe_asset_file = self.humanoid_asset_file
            keyframe_asset = self.gym.load_asset(
                self.sim, self.humanoid_asset_root, keyframe_asset_file, keyframe_asset_options
            )
            num_keyframe_bodies = self.gym.get_asset_rigid_body_count(keyframe_asset)
            num_keyframe_shapes = self.gym.get_asset_rigid_shape_count(keyframe_asset)
            else_assets = [keyframe_asset for _ in range(self.num_keyframes)]
            else_start_poses = [self.humanoid_start_pose for _ in range(self.num_keyframes)]
            else_num_bodies = [num_keyframe_bodies for _ in range(self.num_keyframes)]
            else_num_shapes = [num_keyframe_shapes for _ in range(self.num_keyframes)]

        return else_assets, else_start_poses, else_num_bodies, else_num_shapes

    def _create_else_actors(self, env_ptr, env_idx, else_assets, else_start_poses):
        else_handles = []
        if self._use_keyframes and self.cfg["env"]["test"]:
            contact_filter = (
                1  # >1 : ignore all the collision, 0 : enable all collision, -1 : collision defined by robot file
            )
            for k in range(1, self.num_keyframes + 1):
                segmentation_id = k  # segmentation ID used in segmentation camera sensors
                handle = self.gym.create_actor(
                    env_ptr,
                    else_assets[k - 1],
                    else_start_poses[k - 1],
                    "keyframe_%d" % k,
                    self.num_envs * k + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                for j in range(self.humanoid_num_bodies):
                    self.gym.set_rigid_body_color(
                        env_ptr,
                        handle,
                        j,
                        gymapi.MESH_VISUAL,
                        gymapi.Vec3(1 - k / self.num_keyframes, k / self.num_keyframes, 0.1),
                    )
                else_handles.append(handle)

        return else_handles

    def set_else_actors_dof_properties(self, env_ptr, else_handles, else_assets):
        if self._use_keyframes and self.cfg["env"]["test"]:
            if self._pd_control:
                for handle, asset in zip(else_handles, else_assets):
                    dof_prop = self.gym.get_asset_dof_properties(asset)
                    dof_prop["driveMode"] = gymapi.DOF_MODE_NONE
                    dof_prop["friction"][:] = 1000000
                    dof_prop["damping"][:] = 1000000
                    dof_prop["stiffness"][:] = 1000000
                    self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
        return

    def post_physics_step(self):
        if self._use_keyframes:
            self._update_curr_keyframe()

        self.progress_buf += 1

        if self.cfg["env"]["kinematic"]:
            self._motion_sync()

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def _compute_keyframe_observations(self, env_ids, indices):
        if env_ids is None:
            kf_root_pos = self._kf_root_pos_buf[self.all_env_ids, indices]
            kf_root_rot = self._kf_root_rot_buf[self.all_env_ids, indices]
            kf_dof_pos = self._kf_dof_pos_buf[self.all_env_ids, indices]
            kf_key_pos = self._kf_key_pos_buf[self.all_env_ids, indices]

            sim_root_pos = self._humanoid_root_states[:, :3]
            sim_root_rot = self._humanoid_root_states[:, 3:7]

            pe = self._positional_encoding[self._time_to_arrival - 1]
        else:
            kf_root_pos = self._kf_root_pos_buf[env_ids, indices[env_ids]]
            kf_root_rot = self._kf_root_rot_buf[env_ids, indices[env_ids]]
            kf_dof_pos = self._kf_dof_pos_buf[env_ids, indices[env_ids]]
            kf_key_pos = self._kf_key_pos_buf[env_ids, indices[env_ids]]

            sim_root_pos = self._humanoid_root_states[env_ids, :3]
            sim_root_rot = self._humanoid_root_states[env_ids, 3:7]
            pe = self._positional_encoding[self._time_to_arrival[env_ids] - 1]

        kf_obs = compute_keyframe_observations(
            kf_root_pos, kf_root_rot, kf_dof_pos, kf_key_pos, sim_root_pos, sim_root_rot
        )
        return kf_obs + pe

    def _compute_observations(self, env_ids=None):
        # default simulation states (defined in humanoid_amp_base.py)
        sim_obs = super()._compute_observations(env_ids=env_ids)

        if self._use_keyframes:
            curr_kf_obs = self._compute_keyframe_observations(env_ids, self._curr_keyframe_indices)
            next_kf_obs = self._compute_keyframe_observations(env_ids, self._curr_keyframe_indices + 1)
            if env_ids is None:
                self.obs_buf[:] = torch.cat([sim_obs, curr_kf_obs, next_kf_obs], dim=-1)
            else:
                self.obs_buf[env_ids] = torch.cat([sim_obs, curr_kf_obs, next_kf_obs], dim=-1)
        else:  # when not using keyframes
            if env_ids is None:
                self.obs_buf[:] = sim_obs
            else:
                self.obs_buf[env_ids] = sim_obs

    def _compute_reward(self, actions):
        if self._use_keyframes:
            # joint rotations (global)
            curr_rigid_body_rot = self._humanoid_rigid_body_rot
            goal_rigid_body_rot = self._ref_rigid_body_rot_buf[self.all_env_ids, self.progress_buf]

            # joint velocities (global)
            curr_rigid_body_ang_vel = self._humanoid_rigid_body_ang_vel
            goal_rigid_body_ang_vel = self._ref_rigid_body_ang_vel_buf[self.all_env_ids, self.progress_buf]

            # ee pos (global)
            curr_ee_pos = self._humanoid_rigid_body_pos[:, self._key_body_ids]
            goal_ee_pos = self._ref_key_pos_buf[self.all_env_ids, self.progress_buf]

            # com pos
            curr_root_pos = self._humanoid_root_states[:, :3]
            goal_root_pos = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, :3]

            self.rew_buf[:] = compute_imitation_reward(
                curr_rigid_body_rot,
                goal_rigid_body_rot,
                curr_rigid_body_ang_vel,
                goal_rigid_body_ang_vel,
                curr_ee_pos,
                goal_ee_pos,
                curr_root_pos,
                goal_root_pos,
            )
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

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            dof_body_ids=DOF_BODY_IDS,
            dof_offsets=DOF_OFFSETS,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device,
        )
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidAMP.StateInit.Default:
            self._reset_default(env_ids)
        elif self._state_init == HumanoidAMP.StateInit.Start or self._state_init == HumanoidAMP.StateInit.Random:
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidAMP.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        return

    def _reset_default(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._initial_root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

        self._reset_default_env_ids = env_ids
        return

    def _sample_ref_states(self, motion_ids, motion_times):
        if len(motion_ids.shape) == 1:
            motion_ids = motion_ids[:, None]
            motion_times = motion_times[:, None]
        num_envs, num_ref = motion_ids.shape
        motion_ids = motion_ids.reshape(-1)
        motion_times = motion_times.reshape(-1)

        # sample
        output_global = True if self._use_keyframes else False
        (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
            global_rot,
            global_ang_vel,
        ) = self._motion_lib.get_motion_state(motion_ids, motion_times, output_global=output_global)
        root_pos = root_pos.view(num_envs, num_ref, 3)
        root_rot = root_rot.view(num_envs, num_ref, 4)
        root_vel = root_vel.view(num_envs, num_ref, 3)
        root_ang_vel = root_ang_vel.view(num_envs, num_ref, 3)
        dof_pos = dof_pos.view(num_envs, num_ref, self.humanoid_num_dof)
        dof_vel = dof_vel.view(num_envs, num_ref, self.humanoid_num_dof)
        key_pos = key_pos.view(num_envs, num_ref, self._key_body_ids.shape[0], 3)
        if output_global:
            global_rot = global_rot.view(num_envs, num_ref, self.humanoid_num_bodies, 4)
            global_ang_vel = global_ang_vel.view(num_envs, num_ref, self.humanoid_num_bodies, 3)

        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_pos, global_rot, global_ang_vel

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if self._state_init == HumanoidAMP.StateInit.Random or self._state_init == HumanoidAMP.StateInit.Hybrid:
            if self._use_keyframes:
                motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=self.truncate_time)
                if not self.cfg["env"]["kinematic"]:
                    # (jinseok) if some of motion clips are shorter than truncate time, then we need to set them 0
                    _modify_ids = torch.where(motion_times < 0)
                    motion_times[_modify_ids] = 0
                else:
                    motion_times[:] = 0

                # modify
                motion_ids = torch.stack(
                    [motion_ids] * (self.max_episode_length + 1), axis=1
                )  # (num_envs, self.max_episode_length)
                motion_times = torch.stack(
                    [motion_times + k * self.dt * self.control_freq_inv for k in range(self.max_episode_length + 1)],
                    axis=1,
                )  # (num_envs, self.max_episode_length)
            else:
                motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == HumanoidAMP.StateInit.Start:
            motion_times = torch.zeros(num_envs)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            global_rot,
            global_ang_vel,
        ) = self._sample_ref_states(
            motion_ids,
            motion_times,
        )

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_rot=global_rot,
            global_ang_vel=global_ang_vel,
        )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids[:, 0]
        self._reset_ref_motion_times = motion_times[:, 0]

        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

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

    def _set_humanoid_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._humanoid_dof_pos[env_ids] = dof_pos
        self._humanoid_dof_vel[env_ids] = dof_vel
        return

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
        global_rot,
        global_ang_vel,
    ):
        if self._use_keyframes:
            # set reward buf
            self._ref_root_states_buf[env_ids] = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
            self._ref_dof_pos_buf[env_ids] = dof_pos
            self._ref_dof_vel_buf[env_ids] = dof_vel
            self._ref_key_pos_buf[env_ids] = key_pos
            self._ref_rigid_body_rot_buf[env_ids] = global_rot
            self._ref_rigid_body_ang_vel_buf[env_ids] = global_ang_vel

            # set keyframe obs
            self._time_to_arrival[env_ids] = self._keyframe_interval_in_frames
            self._curr_keyframe_indices[env_ids] = 0

            # for rendering
            if self.cfg["env"]["test"]:  # only for rendering
                # set poses
                self._else_root_states[env_ids, :, 0:3] = root_pos[:, self._keyframe_indices]
                self._else_root_states[env_ids, :, 3:7] = root_rot[:, self._keyframe_indices]
                self._else_dof_pos[env_ids] = dof_pos[:, self._keyframe_indices].view(len(env_ids), -1)
                # zero velocities
                self._else_root_states[env_ids, :, 7:10] = 0
                self._else_root_states[env_ids, :, 10:13] = 0
                self._else_dof_vel[env_ids] = 0
                if self._pd_control:
                    self._target_actions[env_ids, self.humanoid_num_dof :] = self._else_dof_pos[env_ids]
        return

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
        global_rot,
        global_ang_vel,
    ):
        self._set_humanoid_state(
            env_ids=env_ids,
            root_pos=root_pos[:, 0],
            root_rot=root_rot[:, 0],
            dof_pos=dof_pos[:, 0],
            root_vel=root_vel[:, 0],
            root_ang_vel=root_ang_vel[:, 0],
            dof_vel=dof_vel[:, 0],
        )
        self._set_else_states(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            key_pos=key_pos,
            global_rot=global_rot,
            global_ang_vel=global_ang_vel,
        )
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

    def _update_curr_keyframe(self):  # helper function to update curr_keyframe_obs
        self._time_to_arrival -= 1
        for idx in range(1, self.num_keyframes - 1):
            change_ids = torch.where(
                self.progress_buf == idx * self._keyframe_interval_in_frames - 1,
            )
            if len(change_ids[0]) > 0:
                self._curr_keyframe_indices[change_ids] += 1
                self._time_to_arrival[change_ids] = self._keyframe_interval_in_frames
        return

    def _motion_sync(self):
        goal_root_states = self._ref_root_states_buf[self.all_env_ids, self.progress_buf]
        goal_dof_pos = self._ref_dof_pos_buf[self.all_env_ids, self.progress_buf]
        goal_dof_vel = self._ref_dof_vel_buf[self.all_env_ids, self.progress_buf]

        zero_root_vel = torch.zeros_like(goal_root_states[:, 7:10])
        zero_root_ang_vel = torch.zeros_like(goal_root_states[:, 10:])
        zero_dof_vel = torch.zeros_like(goal_dof_vel)

        zero_root_vel = goal_root_states[:, 7:10]
        zero_root_ang_vel = goal_root_states[:, 10:]
        zero_dof_vel = goal_dof_vel

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._set_humanoid_state(
            env_ids=env_ids,
            root_pos=goal_root_states[:, :3],
            root_rot=goal_root_states[:, 3:7],
            dof_pos=goal_dof_pos,
            root_vel=zero_root_vel,
            root_ang_vel=zero_root_ang_vel,
            dof_vel=zero_dof_vel,
        )
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
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

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


# keyframe related jit functions
@torch.jit.script
def compute_keyframe_observations(kf_root_pos, kf_root_rot, kf_dof_pos, kf_key_pos, sim_root_pos, sim_root_rot):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    # heading rots
    sim_heading_rot = calc_heading_quat_inv(sim_root_rot)
    kf_heading_rot = calc_heading_quat_inv(kf_root_rot)

    # root pos
    kf_root_pos_obs = my_quat_rotate(sim_heading_rot, kf_root_pos - sim_root_pos)
    # root rot
    kf_root_rot_obs = quat_mul(sim_heading_rot, kf_root_rot)
    kf_root_rot_obs = quat_to_tan_norm(kf_root_rot_obs)
    # dof pos
    kf_dof_obs = dof_to_obs(kf_dof_pos)
    # key pos
    N, J, _ = kf_key_pos.shape
    kf_root_pos_expand = kf_root_pos.unsqueeze(-2)
    kf_local_key_body_pos = kf_key_pos - kf_root_pos_expand
    kf_heading_rot_expand = kf_heading_rot.unsqueeze(-2)
    kf_heading_rot_expand = kf_heading_rot_expand.repeat((1, J, 1))
    kf_flat_end_pos = kf_local_key_body_pos.view(N * J, 3)
    kf_flat_heading_rot = kf_heading_rot_expand.view(N * J, 4)
    kf_key_pos_obs = my_quat_rotate(kf_flat_heading_rot, kf_flat_end_pos).view(N, -1)

    obs = torch.cat(
        (
            kf_root_pos_obs,
            kf_root_rot_obs,
            kf_dof_obs,
            kf_key_pos_obs,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
# inspired by reference - deepmimic (Peng, 2018)
def compute_imitation_reward(
    curr_rigid_body_rot,
    goal_rigid_body_rot,
    curr_rigid_body_ang_vel,
    goal_rigid_body_ang_vel,
    curr_ee_pos,
    goal_ee_pos,
    curr_root_pos,
    goal_root_pos,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    """
    obs = {
        root_states (13),
        dof_pos (28),
        dof_vel (28),
        key_pos (12)
    }
    """
    # compute errors
    N, J, _ = curr_rigid_body_rot.shape
    curr_rigid_body_rot = curr_rigid_body_rot.reshape(-1, 4)
    goal_rigid_body_rot = goal_rigid_body_rot.reshape(-1, 4)
    body_rot_error = quat_diff_rad(curr_rigid_body_rot, goal_rigid_body_rot).view(N, J).pow(2).sum(dim=-1)

    body_ang_vel_error = (curr_rigid_body_ang_vel - goal_rigid_body_ang_vel).pow(2).sum(dim=(-1, -2))

    ee_pos_error = (curr_ee_pos - goal_ee_pos).pow(2).sum(dim=(-1, -2))

    root_pos_error = (curr_root_pos - goal_root_pos).pow(2).sum(dim=-1)

    # compute reward
    body_rot_reward = torch.exp(-2 * body_rot_error)
    body_ang_vel_reward = torch.exp(-0.1 * body_ang_vel_error)
    ee_pos_reward = torch.exp(-40 * ee_pos_error)
    root_pos_reward = torch.exp(-10 * root_pos_error)

    # deepmimic version
    reward = 0.65 * body_rot_reward + 0.1 * body_ang_vel_reward + 0.15 * ee_pos_reward + 0.1 * root_pos_reward
    # print(body_rot_reward[0].item(), body_ang_vel_reward[0].item(), ee_pos_reward[0].item(), root_pos_reward[0].item())

    return reward
