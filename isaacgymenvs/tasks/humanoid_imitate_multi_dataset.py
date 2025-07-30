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
    KEY_BODY_NAMES,
)
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
    normalize,
    exp_map_to_quat,
    quat_from_angle_axis
)
from isaacgymenvs.utils.smpl_sim.smpl_eval import compute_metrics_lite

PLANE_CONTACT_THRESHOLD = 0.05  # at the initial frame, foot must higher than this value
# Obses
# NUM_OBS - Proprioceptive states (defined in humanoid_amp_base.py)
NUM_NEXT_OBS = 15 * (3 + 6 + 3 + 3)  # right next frame

ET_THRESHOLD = 0.1
PREPARATION_TIME = 0
ET_TIMEWINDOW = 30

# TOTAL BODY NAMES
# pelvis, torso, head, right_upper_arm, right_lower_arm, right_hand, left_upper_arm, left_lower_arm, left_hand, right_thigh, right_shin, right_foot, left_thigh, left_shin, left_foot

UPPER_LOWER_DOF_GROUP = [[i for i in range(14)], [i for i in range(14, 28)]] # upper and lower
UPPER_LOWER_BODY_GROUP = [[i for i in range(1, 9)], [0, ] + [i for i in range(9, 15)]] # upper and lower
UPPER_LOWER_KEY_BODY_GROUP = [[i for i in range(2)], [i for i in range(2, 4)]] # upper and lower
# TRUNK_LIMBS_DOF_GROUP not implemented yet

class HumanoidImitateMultiDataset(HumanoidAMPBase):
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
        
        # dof_group
        self._dof_group = cfg["env"].get("dof_group", "upper-lower")
        if self._dof_group == "upper-lower":
            self._dof_group_ids = UPPER_LOWER_DOF_GROUP
            self._body_group_ids = UPPER_LOWER_BODY_GROUP
            self._key_body_group_ids = UPPER_LOWER_KEY_BODY_GROUP
            self._root_group_id = 1 # lower motion has priority on root states
        else:
            raise NotImplementedError()

        # render
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidImitateMultiDataset.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # goal
        self._num_next_obs_steps = cfg["env"].get("numNextObsSteps", 1)
        self.goal_obs_space = spaces.Box(np.ones(self.num_goal_obs) * -np.Inf, np.ones(self.num_goal_obs) * np.Inf)
        self.clip_goal_obs = self.cfg["env"].get("clipGoalObservations", np.Inf)

        # reward style
        self.imitation_rew_style = self.cfg["env"]["imitationRewStyle"]
        self.imitation_reset_style = self.cfg["env"]["imitationResetStyle"]
        if not self.cfg["env"]["test"]: # train
            try:
                assert not (self.imitation_rew_style == "deepmimic" and not self.cfg["env"]["enableEarlyTermination"])
            except:
                raise ValueError("deepmimic reward is only allowable for training where early termination is enabled!")

        self._energy_rew_coef = self.cfg["env"].get("energyRewCoef", 0.0)

        # visualize reference motion
        self._display_reference = self.cfg["env"]["displayReference"] and self.cfg["env"]["test"] and not headless and not self.cfg["env"].get("prior_rollout", False)

        # motion sampling related
        fps = round(1 / (self.cfg["sim"]["dt"] + 1e-7))  # assume fps is always integer
        control_freq_inv = self.cfg["env"]["controlFrequencyInv"]
        self.max_episode_length_in_time = max_episode_length_in_time = self.cfg["env"]["episodeLength"] / (
            fps / control_freq_inv
        )  # in seconds (60 / ((1/60) / 2))
        self.truncate_time = self.cfg["env"].get("truncateTime", max_episode_length_in_time)  # use for sampling reference motion

        # render
        self.cfg["env"]["renderFPS"] = 60
        if self.cfg["env"]["kinematic"]:
            self.ori_control_freq_inv = self.cfg["env"]["controlFrequencyInv"]
            self.cfg["env"]["controlFrequencyInv"] = 1
            self.cfg["env"]["renderFPS"] = 30
            self._enable_early_termination = False
        
        self.render_every = max(fps // self.cfg["env"]["renderFPS"], 1)

        # eval - prior rollout
        self.cfg["env"]["prior_rollout"] = False if not self.cfg["env"]["test"] else self.cfg["env"].get("prior_rollout", False)
        if self.cfg["env"]["prior_rollout"]:
            self.cfg["env"]["envSpacing"] = 0
            self.cfg["env"]["enableEarlyTermination"] = False

        # eval - jittering test
        self.eval_jitter = cfg["env"].get("eval_jitter", False) if cfg["env"]["test"] else False
        self.eval_metric = self.cfg["env"].get("eval_metric", False)
        if self.eval_metric:
            self.cfg["env"]["enableEarlyTermination"] = False

        # data extraction
        self.data_extraction = self.cfg["env"].get("data_extraction", False)
        if self.data_extraction:
            self.cfg["env"]["enableEarlyTermination"] = False
            assert self.cfg["env"]["episodeLength"] % 30 == 0
            self.cfg["env"]["episodeLength"] += 1

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        motion_file_list = cfg["env"].get("motion_file", ["amp_humanoid_backflip.npy"])
        motion_file_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/motions/")

        self._low_memory_load = cfg["env"].get("low_memory_load", False)
        self._noise_scale = cfg["env"].get("noise_scale", 0.0)
        self._add_noise = True if self._noise_scale > 0 else False

        motion_file_paths = []
        for motion_file in motion_file_list:
            motion_file_path = []
            if isinstance(motion_file, str):
                motion_file = os.path.join(motion_file_root_path, motion_file)
                # Case 1 : if it is single motion file
                if motion_file.split(".")[-1] == "npy":
                    motion_file_path.append(motion_file)
                # Case 2 : if it is directory
                elif os.path.isdir(motion_file):
                    temp_motion_file_path = list(Path(motion_file).rglob("*.npy"))
                    if "LaFAN1" in motion_file:
                        if hasattr(self, "exclude_motion_keys"):
                            for exclude_key in self.exclude_motion_keys:
                                temp_motion_file_path = [path for path in temp_motion_file_path if exclude_key not in str(path)]
                        else:
                            temp_motion_file_path = [path for path in temp_motion_file_path if 'obstacles' not in str(path)]
                    motion_file_path.extend(temp_motion_file_path)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
            motion_file_paths.append(motion_file_path)

        self._load_motion(motion_file_paths)

        # foot shape
        foot_offset = torch.tensor(
            [0.045, 0, -0.0255], device=self.device, dtype=torch.float32, requires_grad=False
        )  # offset from foot joint to the rigid body com (box)
        query_points = torch.tensor(
            [
                [0.0895, 0.045, 0.0285],
                [0.0895, 0.045, -0.0285],
                [0.0895, -0.045, 0.0285],
                [0.0895, -0.045, -0.0285],
                [-0.0895, 0.045, 0.0285],
                [-0.0895, 0.045, -0.0285],
                [-0.0895, -0.045, 0.0285],
                [-0.0895, -0.045, -0.0285],
            ],
            device=self.device,
            dtype=torch.float32,
            requires_grad=False,
        )
        self.query_points = foot_offset[None] + query_points  # (offset from foot joint for each query points)

        # Return Buffer
        self.return_buf = torch.zeros(
            self.num_envs, self.max_episode_length + 1, device=self.device, dtype=torch.float32
        )

        # Reference Motion Buffer
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)
        self._ref_buf_length = self.max_episode_length + self._num_next_obs_steps
        self._ref_root_states_buf = torch.zeros(
            self.num_envs, self._ref_buf_length, 13, device=self.device, dtype=torch.float32
        )
        self._ref_dof_pos_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_dof,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_dof_vel_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_dof,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_key_pos_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self._key_body_ids.shape[0],
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_rigid_body_pos_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_bodies,
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_rigid_body_rot_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_bodies,
            4,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_rigid_body_vel_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_bodies,
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._ref_rigid_body_ang_vel_buf = torch.zeros(
            self.num_envs,
            self._ref_buf_length,
            self.humanoid_num_bodies,
            3,
            device=self.device,
            dtype=torch.float32,
        )
        self._curr_motion_ids_all = []
        for _ in range(len(self._motion_libs)):
            _curr_motion_ids = torch.zeros(
                self.num_envs,
                device=self.device,
                dtype=torch.int64,
            )
            self._curr_motion_ids_all.append(_curr_motion_ids)

        self._updating_motion_weight = False

        if self._display_reference and hasattr(self, "_ref_actor_idx"):
            if self.cfg["env"]["kinematic"]:
                self.ref_actor_indices = self.global_actor_indices[:, 0].clone()
                self.ref_dof_actor_indices = self.global_dof_actor_indices[:, 0].clone()
            else:
                self.ref_actor_indices = self.global_actor_indices[:, 1 + self._ref_actor_idx].clone()
                self.ref_dof_actor_indices = self.global_dof_actor_indices[:, 1 + self._ref_dof_start_idx].clone()
        return

    def _create_ground_plane(self):
        if not self._display_reference:
            plane_params = gymapi.PlaneParams()
            plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
            plane_params.static_friction = self.plane_static_friction
            plane_params.dynamic_friction = self.plane_dynamic_friction
            plane_params.restitution = self.plane_restitution
            self.gym.add_ground(self.sim, plane_params)
        else: # use box instead of ground plane (to disable collision to reference bodies)
            BOX_X, BOX_Y, BOX_Z = 100, 100, 0.2
            box_dims = gymapi.Vec3(BOX_X, BOX_Y, BOX_Z)
            box_asset_options = gymapi.AssetOptions()
            box_asset_options.fix_base_link = True
            # box_asset_options.collapse_fixed_joints = True
            box_asset_options.disable_gravity = True
            box_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, box_asset_options)
            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(0, 0, -BOX_Z / 2)
            box_prop = self.gym.get_asset_rigid_shape_properties(box_asset)
            box_prop[0].friction = 1.0
            box_prop[0].rolling_friction = box_prop[0].friction / 100.0
            self.gym.set_asset_rigid_shape_properties(box_asset, box_prop)
            self.ground_asset = box_asset
            self.ground_start_pose = box_pose
        return

    def _prepare_else_assets(self):
        # NOTE THAT HUMANOID has priority
        # Ex) MUST DEFINE HUMANOID HANDLE AND THAN OBJECTS
        else_assets, else_start_poses, else_num_bodies, else_num_shapes = [], [], [], []
        self.num_else_actor = 0
        # ground plane
        if hasattr(self, "ground_asset"):
            else_assets.append(self.ground_asset)
            else_start_poses.append(self.ground_start_pose)
            else_num_bodies.append(1)
            else_num_shapes.append(1)
            self.num_else_actor += 1

        if self._display_reference and not self.cfg["env"]["kinematic"]:
            humanoid_asset_options = gymapi.AssetOptions()
            humanoid_asset_file = self.humanoid_asset_file
            # motion reference actor
            humanoid_asset_options.fix_base_link = False
            humanoid_asset_options.disable_gravity = True
            ref_humanoid_asset = self.gym.load_asset(
                self.sim, self.humanoid_asset_root, humanoid_asset_file, humanoid_asset_options
            )
            else_assets.append(ref_humanoid_asset)
            else_start_poses.append(self.humanoid_start_pose)
            else_num_bodies.append(self.gym.get_asset_rigid_body_count(ref_humanoid_asset))
            else_num_shapes.append(self.gym.get_asset_rigid_shape_count(ref_humanoid_asset))
            self.num_else_actor += 1

        return else_assets, else_start_poses, else_num_bodies, else_num_shapes

    def _create_else_actors(self, env_ptr, env_idx, else_assets, else_start_poses):
        else_handles = []
        # ground plane
        if hasattr(self, "ground_asset"):
            contact_filter = 0
            segmentation_id = 1
            handle = self.gym.create_actor(
                env_ptr,
                else_assets[0],
                else_start_poses[0],
                "ground",
                env_idx,
                contact_filter,
                segmentation_id,
            )
            ground_color = gymapi.Vec3(0.3, 0.3, 0.3)
            self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, ground_color)
            else_handles.append(handle)

        # agent
        if self._display_reference and not self.cfg["env"]["kinematic"]:
            offset = 1 if hasattr(self, "ground_asset") else 0
            contact_filter = (
                1  # >1 : ignore all the collision, 0 : enable all collision, -1 : collision defined by robot file
            )
            self._ref_actor_idx = 1  # index among else actors
            self._ref_dof_start_idx = 0

            # handle for ref agent
            segmentation_id = 2 # segmentation ID used in segmentation camera sensors
            handle = self.gym.create_actor(
                env_ptr,
                else_assets[offset],
                else_start_poses[offset],
                "reference",
                self.num_envs + env_idx,
                contact_filter,
                segmentation_id,
            )
            humanoid_color = gymapi.Vec3(0.8, 0.8, 0.8)
            for j in range(self.humanoid_num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL, humanoid_color)
            else_handles.append(handle)

        return else_handles

    def set_else_actors_dof_properties(self, env_ptr, else_handles, else_assets):
        if self._display_reference:
            if self._pd_control:
                for handle, asset in zip(else_handles, else_assets):
                    dof_prop = self.gym.get_asset_dof_properties(asset)
                    dof_prop["driveMode"] = gymapi.DOF_MODE_NONE
                    dof_prop["friction"][:] = 1000000
                    dof_prop["damping"][:] = 0
                    dof_prop["stiffness"][:] = 0
                    self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
        return

    # NCP style action to pd target
    def _action_to_pd_targets(self, action):
        pd_tar = self._humanoid_dof_pos + self._pd_action_scale * action
        # clamp
        pd_tar = torch.maximum(pd_tar, self._pd_action_limit_lower)
        pd_tar = torch.minimum(pd_tar, self._pd_action_limit_upper)
        return pd_tar

    def post_physics_step(self):
        self.progress_buf += 1

        if not self.cfg["env"]["kinematic"]:
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
        
        # data extraction
        if self.data_extraction:
            self.store_sim_states(query=["root", "dof_pos", "dof_vel", "local_ee_pos", "local_ee_contacts", "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel"])
        return

    def _compute_ref_diff_observations(self, env_ids=None):
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

        if self._add_noise:
            scale = self._noise_scale
            ref_rigid_body_pos += torch.empty_like(ref_rigid_body_pos).normal_(generator=self.env_rng) * scale
            # ref_rigid_body_rot += torch.empty_like(ref_rigid_body_rot).normal_(generator=self.env_rng) * scale
            # ref_rigid_body_rot = normalize(ref_rigid_body_rot)
            ref_rigid_body_vel += torch.empty_like(ref_rigid_body_vel).normal_(generator=self.env_rng) * scale
            # ref_rigid_body_ang_vel += torch.empty_like(ref_rigid_body_ang_vel).normal_(generator=self.env_rng) * scale

        J = self.humanoid_num_bodies
        ref_diff_obs = compute_ref_diff_observations(
            ref_rigid_body_pos.view(-1, J, 3),
            ref_rigid_body_rot.view(-1, J, 4),
            ref_rigid_body_vel.view(-1, J, 3),
            ref_rigid_body_ang_vel.view(-1, J, 3),
            sim_rigid_body_pos.view(-1, J, 3),
            sim_rigid_body_rot.view(-1, J, 4),
            sim_rigid_body_vel.view(-1, J, 3),
            sim_rigid_body_ang_vel.view(-1, J, 3),
            sim_root_rot.view(-1, 4),
        )
        ref_diff_obs = ref_diff_obs.view(B, -1)
        return ref_diff_obs

    def _compute_observations(self, env_ids=None):
        # default simulation states (defined in humanoid_amp_base.py)
        sim_obs = super()._compute_observations(env_ids=env_ids)
        goal_obs = self._compute_goal_observations(env_ids=env_ids)

        if env_ids is None:
            self.obs_buf[:] = sim_obs
            if goal_obs != None:
                self.goal_obs_buf[:] = goal_obs
        else:
            self.obs_buf[env_ids] = sim_obs
            if goal_obs != None:
                self.goal_obs_buf[env_ids] = goal_obs

    def _compute_goal_observations(self, env_ids=None):
        goal_obs = self._compute_ref_diff_observations(env_ids)
        return goal_obs

    def _compute_reward(self, actions):
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

        # ee pos (global)
        curr_ee_pos = self._humanoid_rigid_body_pos[:, self._key_body_ids]
        goal_ee_pos = self._ref_key_pos_buf[self.all_env_ids, self.progress_buf]

        # joint rotations (local)
        curr_dof_pos = self._humanoid_dof_pos
        goal_dof_pos = self._ref_dof_pos_buf[self.all_env_ids, self.progress_buf]

        # joint angular velocities (local)
        curr_dof_vel = self._humanoid_dof_vel
        goal_dof_vel = self._ref_dof_vel_buf[self.all_env_ids, self.progress_buf]

        # com pos
        curr_root_pos = self._humanoid_root_states[:, :3]
        goal_root_pos = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, :3]
        
        # com rot
        curr_root_rot = self._humanoid_root_states[:, 3:7]
        goal_root_rot = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, 3:7]

        # com vel
        curr_root_vel = self._humanoid_root_states[:, 7:10]
        goal_root_vel = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, 7:10]

        # com ang vel
        curr_root_ang_vel = self._humanoid_root_states[:, 10:]
        goal_root_ang_vel = self._ref_root_states_buf[self.all_env_ids, self.progress_buf, 10:]

        if self.imitation_rew_style in ["deepmimic", "deepmimic_mul"]:
            if self.imitation_rew_style == "deepmimic_mul":
                multiplication = True
            else:
                multiplication = False
            self.rew_buf[:] = compute_deepmimic_reward(
                curr_dof_pos,
                goal_dof_pos,
                curr_dof_vel,
                goal_dof_vel,
                curr_ee_pos,
                goal_ee_pos,
                curr_root_pos,
                goal_root_pos,
                multiplication,
            )
        elif self.imitation_rew_style == "phc":
            self.rew_buf[:] = compute_phc_reward(
                curr_rigid_body_pos,
                goal_rigid_body_pos,
                curr_rigid_body_rot,
                goal_rigid_body_rot,
                curr_rigid_body_vel,
                goal_rigid_body_vel,
                curr_rigid_body_ang_vel,
                goal_rigid_body_ang_vel,
            )
        
        if self._energy_rew_coef > 0:
            energy_rew = compute_energy_reward(
                actions=actions,
                dof_vel=curr_dof_vel,
                coef=self._energy_rew_coef
            )
            self.rew_buf = 0.95 * self.rew_buf + 0.05 * energy_rew

        self.return_buf[self.all_env_ids, self.progress_buf] = self.rew_buf

        return

    def _compute_reset(self, early_termination=None):
        if early_termination is None:
            early_termination = self._enable_early_termination

        curr_rigid_body_pos = self._humanoid_rigid_body_pos
        goal_rigid_body_pos = self._ref_rigid_body_pos_buf[self.all_env_ids, self.progress_buf]
        if self.imitation_reset_style == "normal":
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
                self.reset_buf,
                self.progress_buf,
                self._humanoid_contact_forces,
                self._contact_body_ids,
                self._humanoid_rigid_body_pos,
                self.max_episode_length,
                early_termination,
                self._termination_height,
            )
        elif self.imitation_reset_style == "reward":
            self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
                self.reset_buf,
                self.progress_buf,
                self._humanoid_contact_forces,
                self._contact_body_ids,
                self._humanoid_rigid_body_pos,
                self.max_episode_length,
                False,
                self._termination_height,
            )
            env_ids = (1 - self.reset_buf).nonzero(as_tuple=False).squeeze(-1)
            self.return_buf[:, : PREPARATION_TIME + 1] = 0
            if len(env_ids) > 0 and early_termination:
                time_window = torch.stack([self.progress_buf[env_ids] - i for i in range(ET_TIMEWINDOW)], dim=-1)
                windowed_return = self.return_buf[env_ids[:, None], time_window].sum(dim=-1)
                windowed_time = torch.minimum(
                    self.progress_buf[env_ids] - PREPARATION_TIME,
                    torch.ones_like(self.progress_buf[env_ids]) * ET_TIMEWINDOW,
                )
                terminated = windowed_return < ET_THRESHOLD * windowed_time
                terminated = torch.logical_and(terminated, self.progress_buf[env_ids] > PREPARATION_TIME)
                self._terminate_buf[env_ids] = terminated.long()
                self.reset_buf[env_ids] = torch.where(
                    terminated, torch.ones_like(self.reset_buf[env_ids]), self.reset_buf[env_ids]
                )

        elif self.imitation_reset_style == "error_max":
            trigger = (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).pow(0.5).max(dim=-1).values[0].item()
            self.reset_buf[:], self._terminate_buf[:] = compute_imitation_reset_max(
                self.reset_buf,
                self.progress_buf,
                curr_rigid_body_pos,
                goal_rigid_body_pos,
                self._contact_body_ids,
                self.max_episode_length,
                early_termination,
            )
        elif self.imitation_reset_style == "error_mean":
            self.reset_buf[:], self._terminate_buf[:] = compute_imitation_reset_mean(
                self.reset_buf,
                self.progress_buf,
                curr_rigid_body_pos,
                goal_rigid_body_pos,
                self._contact_body_ids,
                self.max_episode_length,
                early_termination,
            )
        elif self.imitation_reset_style == "root_dist":
            self.reset_buf[:], self._terminate_buf[:] = compute_imitation_reset_root_dist(
                self.reset_buf,
                self.progress_buf,
                curr_rigid_body_pos[:, 0],
                goal_rigid_body_pos[:, 0],
                self.max_episode_length,
                early_termination,
            )
        return

    @property
    def num_goal_obs(self) -> int:
        return NUM_NEXT_OBS * self._num_next_obs_steps

    @property
    def goal_observation_space(self) -> gym.Space:
        return self.goal_obs_space

    def allocate_buffers(self):
        super().allocate_buffers()
        self.goal_obs_buf = torch.zeros((self.num_envs, self.num_goal_obs), device=self.device, dtype=torch.float)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

        # randomize actions
        if self.dr_randomizations.get("actions", None):
            actions = self.dr_randomizations["actions"]["noise_lambda"](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            if self.force_render:
                if (self._display_reference or self.cfg["env"]["kinematic"]) and i == 0:
                    self._motion_sync()
                if i % self.render_every == 0:
                    self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        if self.force_render and (self._display_reference or self.cfg["env"]["kinematic"]):
            self.reset_call_flag = False

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get("observations", None):
            self.obs_buf = self.dr_randomizations["observations"]["noise_lambda"](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        # goal related
        self.obs_dict["goal_obs"] = torch.clamp(
            self.goal_obs_buf,
            -self.clip_goal_obs,
            self.clip_goal_obs,
        )
        return (
            self.obs_dict,
            self.rew_buf.to(self.rl_device),
            self.reset_buf.to(self.rl_device),
            self.extras,
        )

    def reset(self):
        _ = super().reset()

        self.obs_dict["goal_obs"] = torch.clamp(
            self.goal_obs_buf,
            -self.clip_goal_obs,
            self.clip_goal_obs,
        )
        return self.obs_dict

    def reset_done(self):
        _, done_env_ids = super().reset_done()
        self.obs_dict["goal_obs"] = torch.clamp(
            self.goal_obs_buf,
            -self.clip_goal_obs,
            self.clip_goal_obs,
        )
        return self.obs_dict, done_env_ids

    def _load_motion(self, motion_files):
        if self.cfg["env"]["test"]: # to faithfully eval
            self.env_rng = torch.Generator(device=self.device)
            self.env_rng.manual_seed(self.cfg["env"]["seed"])
        else:
            self.env_rng = None

        self._motion_libs = []
        for motion_file in motion_files:
            _motion_lib = MotionLib(
                motion_file=motion_file,
                dof_body_ids=DOF_BODY_IDS,
                dof_offsets=DOF_OFFSETS,
                key_body_ids=self._key_body_ids.cpu().numpy(),
                device=self.device,
                min_len=self.max_episode_length_in_time,
                motion_matching=self.cfg["env"].get("motion_matching", False), # eval
                generator=self.env_rng
            )
            self._motion_libs.append(_motion_lib)
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.return_buf[env_ids] = 0
        if self.force_render and (self._display_reference or self.cfg["env"]["kinematic"]):
            self.reset_call_flag = True
        return

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidImitateMultiDataset.StateInit.Default:
            self._reset_default(env_ids)
        elif (
            self._state_init == HumanoidImitateMultiDataset.StateInit.Start or self._state_init == HumanoidImitateMultiDataset.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidImitateMultiDataset.StateInit.Hybrid:
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

    def _sample_ref_states(self, motion_ids_all, motion_times_all):

        root_poses = []
        root_rots = []
        root_vels = []
        root_ang_vels = []
        dof_poses = []  
        dof_vels = []
        key_poses = []
        global_poses = []
        global_rots = []
        global_vels = []
        global_ang_vels = []
        
        group_id = 0
        for motion_ids, motion_times, motion_lib in zip(motion_ids_all, motion_times_all, self._motion_libs):
            if len(motion_ids.shape) == 1:
                motion_ids = motion_ids[:, None]
                motion_times = motion_times[:, None]
            num_envs, num_ref = motion_ids.shape
            motion_ids = motion_ids.reshape(-1)
            motion_times = motion_times.reshape(-1)

            # sample
            (
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
            ) = motion_lib.get_motion_state(motion_ids, motion_times, output_global=True)

            root_poses.append(root_pos.view(num_envs, num_ref, 3))
            root_rots.append(root_rot.view(num_envs, num_ref, 4))
            root_vels.append(root_vel.view(num_envs, num_ref, 3))
            root_ang_vels.append(root_ang_vel.view(num_envs, num_ref, 3))
            dof_poses.append(dof_pos.view(num_envs, num_ref, self.humanoid_num_dof))
            dof_vels.append(dof_vel.view(num_envs, num_ref, self.humanoid_num_dof))
            key_poses.append(key_pos.view(num_envs, num_ref, self._key_body_ids.shape[0], 3))
            global_poses.append(global_pos.view(num_envs, num_ref, self.humanoid_num_bodies, 3))
            global_rots.append(global_rot.view(num_envs, num_ref, self.humanoid_num_bodies, 4))
            global_vels.append(global_vel.view(num_envs, num_ref, self.humanoid_num_bodies, 3))
            global_ang_vels.append(global_ang_vel.view(num_envs, num_ref, self.humanoid_num_bodies, 3))
        
            group_id += 1
            
        """Choose Root First"""
        mod_root_pos = root_poses[self._root_group_id]
        mod_root_vel = root_vels[self._root_group_id]
        mod_root_rot = root_rots[self._root_group_id]
        mod_root_ang_vel = root_ang_vels[self._root_group_id]

        mod_dof_pos = torch.zeros_like(dof_poses[0])
        mod_dof_vel = torch.zeros_like(dof_vels[0])
        mod_key_pos = torch.zeros_like(key_poses[0])
        mod_global_pos = torch.zeros_like(global_poses[0])
        mod_global_rot = torch.zeros_like(global_rots[0])
        mod_global_vel = torch.zeros_like(global_vels[0])
        mod_global_ang_vel = torch.zeros_like(global_ang_vels[0])

        """Synchonize Root with all the body parts"""
        for group_id in range(len(self._motion_libs)):
            mod_dof_pos[:, :, self._dof_group_ids[group_id]] = dof_poses[group_id][:, :, self._dof_group_ids[group_id]]
            mod_dof_vel[:, :, self._dof_group_ids[group_id]] = dof_vels[group_id][:, :, self._dof_group_ids[group_id]]

            if group_id == self._root_group_id:
                mod_key_pos[:, :, self._key_body_group_ids[group_id]] = key_poses[group_id][:, :, self._key_body_group_ids[group_id]]
                mod_global_pos[:, :, self._body_group_ids[group_id]] = global_poses[group_id][:, :, self._body_group_ids[group_id]]
                mod_global_rot[:, :, self._body_group_ids[group_id]] = global_rots[group_id][:, :, self._body_group_ids[group_id]]
                mod_global_vel[:, :, self._body_group_ids[group_id]] = global_vels[group_id][:, :, self._body_group_ids[group_id]] 
                mod_global_ang_vel[:, :, self._body_group_ids[group_id]] = global_ang_vels[group_id][:, :, self._body_group_ids[group_id]]
                continue

            curr_root_rot_inv = quat_conjugate(root_rots[group_id])
            curr_root_pos = root_poses[group_id]
            curr_root_vel = root_vels[group_id]

            # key pose
            _key_pos = key_poses[group_id][:, :, self._key_body_group_ids[group_id]]
            _key_pos = _key_pos - curr_root_pos[:, :, None]
            curr_root_rot_inv_expand = curr_root_rot_inv[:, :, None].repeat(1, 1, _key_pos.shape[2], 1)
            _key_pos = my_quat_rotate(curr_root_rot_inv_expand.view(-1, 4), _key_pos.view(-1, 3)).view(_key_pos.shape)
            target_root_rot_expand = mod_root_rot[:, :, None].repeat(1, 1, _key_pos.shape[2], 1)
            _key_pos = my_quat_rotate(target_root_rot_expand.view(-1, 4), _key_pos.view(-1, 3)).view(_key_pos.shape)
            _key_pos = _key_pos + mod_root_pos[:, :, None]
            mod_key_pos[:, :, self._key_body_group_ids[group_id]] = _key_pos

            # global pose
            _global_pos = global_poses[group_id][:, :, self._body_group_ids[group_id]]
            _global_pos = _global_pos - curr_root_pos[:, :, None]
            curr_root_rot_inv_expand = curr_root_rot_inv[:, :, None].repeat(1, 1, _global_pos.shape[2], 1)
            _global_pos = my_quat_rotate(curr_root_rot_inv_expand.view(-1, 4), _global_pos.view(-1, 3)).view(_global_pos.shape)
            target_root_rot_expand = mod_root_rot[:, :, None].repeat(1, 1, _global_pos.shape[2], 1)
            _global_pos = my_quat_rotate(target_root_rot_expand.view(-1, 4), _global_pos.view(-1, 3)).view(_global_pos.shape)
            _global_pos = _global_pos + mod_root_pos[:, :, None]
            mod_global_pos[:, :, self._body_group_ids[group_id]] = _global_pos

            # global vel
            _global_vel = global_vels[group_id][:, :, self._body_group_ids[group_id]]
            _global_vel = _global_vel - curr_root_vel[:, :, None]
            # curr_root_rot_inv_expand = curr_root_rot_inv[:, :, None].repeat(1, 1, _global_vel.shape[2], 1)
            _global_vel = my_quat_rotate(curr_root_rot_inv_expand.view(-1, 4), _global_vel.view(-1, 3)).view(_global_vel.shape)
            # target_root_rot_expand = root_rot[:, :, None].repeat(1, 1, _global_vel.shape[2], 1)
            _global_vel = my_quat_rotate(target_root_rot_expand.view(-1, 4), _global_vel.view(-1, 3)).view(_global_vel.shape)
            _global_vel = _global_vel + mod_root_vel[:, :, None]
            mod_global_vel[:, :, self._body_group_ids[group_id]] = _global_vel

            # global rot
            _global_rot = global_rots[group_id][:, :, self._body_group_ids[group_id]]
            _global_rot = quat_mul(curr_root_rot_inv_expand.view(-1, 4), _global_rot.view(-1, 4)).view(_global_rot.shape)
            _global_rot = quat_mul(target_root_rot_expand.view(-1, 4), _global_rot.view(-1, 4)).view(_global_rot.shape)
            mod_global_rot[:, :, self._body_group_ids[group_id]] = _global_rot

            # global ang vel
            mod_global_ang_vel[:, :, self._body_group_ids[group_id]] = global_ang_vels[group_id][:, :, self._body_group_ids[group_id]]

        return (
            mod_root_pos,
            mod_root_rot,
            mod_root_vel,
            mod_root_ang_vel,
            mod_dof_pos,
            mod_dof_vel,
            mod_key_pos,
            mod_global_pos,
            mod_global_rot,
            mod_global_vel,
            mod_global_ang_vel,
        )

    # def update_motion_weights(self, motion_values, returns, decay=0.95):
    #     raise NotImplementedError()

    def _sample_motion_ids_and_times(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids_all = []
        motion_times_all = []
        for lib_id, motion_lib in enumerate(self._motion_libs):
            """Motion IDs"""
            motion_ids = motion_lib.sample_motions(num_envs)
            self._curr_motion_ids_all[lib_id][env_ids] = motion_ids

            """Motion Times"""
            if self._state_init == HumanoidImitateMultiDataset.StateInit.Random or self._state_init == HumanoidImitateMultiDataset.StateInit.Hybrid:
                motion_times = motion_lib.sample_time(motion_ids, truncate_time=self.truncate_time)
                if not self.cfg["env"]["kinematic"] and self.cfg["env"]["start_frame"][0] == -1:
                    # (jinseok) if some of motion clips are shorter than truncate time, then we need to set them 0
                    _modify_ids = torch.where(motion_times < 0)
                    motion_times[_modify_ids] = 0
                elif self.cfg["env"]["test"] and self.cfg["env"]["start_frame"][lib_id] > 0:
                    motion_times[:] = self.cfg["env"]["start_frame"][lib_id] * self.dt
                else:
                    motion_times[:] = 0

                # modify
                motion_ids = torch.stack(
                    [motion_ids] * (self._ref_buf_length), axis=1
                )  # (num_envs, self.max_episode_length)
                if self.cfg["env"]["kinematic"]:
                    interval = self.dt * self.ori_control_freq_inv
                else:
                    interval = self.dt
                motion_times = torch.stack(
                    [motion_times + k * interval for k in range(self._ref_buf_length)],
                    axis=1,
                )  # (num_envs, self.max_episode_length)
            elif self._state_init == HumanoidImitateMultiDataset.StateInit.Start:
                motion_times = torch.zeros(num_envs)
            else:
                assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
            motion_ids_all.append(motion_ids)
            motion_times_all.append(motion_times)

        return motion_ids_all, motion_times_all

    def _reset_ref_state_init(self, env_ids):
        motion_ids_all, motion_times_all = self._sample_motion_ids_and_times(env_ids)

        (
            root_pos,
            root_rot,
            root_vel,
            root_ang_vel,
            dof_pos,
            dof_vel,
            key_pos,
            global_pos,
            global_rot,
            global_vel,
            global_ang_vel,
        ) = self._sample_ref_states(
            motion_ids_all,
            motion_times_all,
        )

        if not self.eval_jitter:
            # # eliminate root offset - global joints
            min_foot_pos = global_pos[:, :, [11, 14], 2].min(dim=-1).values.min(dim=-1).values # (B, )
            foot_offset = min_foot_pos - PLANE_CONTACT_THRESHOLD
            foot_offset = torch.maximum(foot_offset, torch.zeros_like(foot_offset))
            root_pos[..., 2] -= foot_offset[:, None]
            global_pos[..., 2] -= foot_offset[:, None, None]
            key_pos[..., 2] -= foot_offset[:, None, None]

            # avoid contact for the first frame
            min_body_pos = global_pos[:, 0, :, 2].min(dim=-1).values # (B,)
            body_offset = min_body_pos - PLANE_CONTACT_THRESHOLD
            body_offset = torch.minimum(body_offset, torch.zeros_like(body_offset))
            root_pos[..., 2] -= body_offset[:, None]
            global_pos[..., 2] -= body_offset[:, None, None]
            key_pos[..., 2] -= body_offset[:, None, None]
        else:
            # zero velocity
            root_vel[:] = 0
            root_ang_vel[:] = 0
            dof_vel[:] = 0
            global_vel[:] = 0
            global_ang_vel[:] = 0
        
        # if self.cfg["env"]["test"] and self.cfg["env"]["white_mode"]:
        #     global_pos[..., :2] -= root_pos[:, 0, None, None, :2]
        #     key_pos[..., :2] -= root_pos[:, 0, None, None, :2]
        #     root_pos[..., :2] -= root_pos[:, 0, None, :2]

        self._set_env_state(
            env_ids=env_ids,
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

        self._reset_ref_env_ids = env_ids

        if self._low_memory_load and self.cfg["env"]["test"]:
            idx = torch.randint(low=0, high=len(self.motion_file_path), size=(1,), generator=self.env_rng_cpu).item()
            self._load_motion(self.motion_file_path[idx:idx+1])
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

    def _set_humanoid_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rigid_body_pos, rigid_body_rot, rigid_body_vel, rigid_body_ang_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self._humanoid_dof_pos[env_ids] = dof_pos
        self._humanoid_dof_vel[env_ids] = dof_vel

        self._humanoid_rigid_body_pos[env_ids] = rigid_body_pos
        self._humanoid_rigid_body_rot[env_ids] = rigid_body_rot
        self._humanoid_rigid_body_vel[env_ids] = rigid_body_vel
        self._humanoid_rigid_body_ang_vel[env_ids] = rigid_body_ang_vel
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
        global_pos,
        global_rot,
        global_vel,
        global_ang_vel,
    ):
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
        global_pos,
        global_rot,
        global_vel,
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
            rigid_body_pos=global_pos[:, 0],
            rigid_body_rot=global_rot[:, 0],
            rigid_body_vel=global_vel[:, 0],
            rigid_body_ang_vel=global_ang_vel[:, 0],
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
            global_pos=global_pos,
            global_rot=global_rot,
            global_vel=global_vel,
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

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        
        if self.cfg["env"]["prior_rollout"]:
            # upper view (prior_viz)
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 8.0, 5)
        else:
            # default
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)

        # static
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        if self.cfg["env"]["prior_rollout"]:
            char_root_pos = self._humanoid_root_states[:, 0:3].mean(dim=0).cpu().numpy()
        else:
            char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _motion_sync(self):
        frame_idx = self.progress_buf + 1
        goal_root_states = self._ref_root_states_buf[self.all_env_ids, frame_idx]
        goal_root_pos = goal_root_states[:, :3]
        goal_root_rot = goal_root_states[:, 3:7]
        goal_root_vel = goal_root_states[:, 7:10]
        goal_root_ang_vel = goal_root_states[:, 10:]
        goal_dof_pos = self._ref_dof_pos_buf[self.all_env_ids, frame_idx]
        goal_dof_vel = self._ref_dof_vel_buf[self.all_env_ids, frame_idx]

        # # offset for motion
        global_pos = self._ref_rigid_body_pos_buf[self.all_env_ids, frame_idx]
        global_rot = self._ref_rigid_body_rot_buf[self.all_env_ids, frame_idx]
        global_vel = self._ref_rigid_body_vel_buf[self.all_env_ids, frame_idx]
        global_ang_vel = self._ref_rigid_body_ang_vel_buf[self.all_env_ids, frame_idx]

        env_ids = self.all_env_ids
        if self.cfg["env"]["kinematic"]:  # kinematic-only
            self._set_humanoid_state(
                env_ids=env_ids,
                root_pos=goal_root_pos,
                root_rot=goal_root_rot,
                dof_pos=goal_dof_pos,
                root_vel=goal_root_vel,
                root_ang_vel=goal_root_ang_vel,
                dof_vel=goal_dof_vel,
                rigid_body_pos=global_pos,
                rigid_body_rot=global_rot,
                rigid_body_vel=global_vel,
                rigid_body_ang_vel=global_ang_vel,
            )
        elif self._display_reference:  # agent & kinematic
            rs_idx = self._ref_actor_idx
            re_idx = rs_idx + self.humanoid_num_bodies
            ds_idx = self._ref_dof_start_idx
            de_idx = ds_idx + self.humanoid_num_dof 
            self._else_root_states[env_ids, self._ref_actor_idx, 0:3] = goal_root_pos
            self._else_root_states[env_ids, self._ref_actor_idx, 3:7] = goal_root_rot
            self._else_root_states[env_ids, self._ref_actor_idx, 7:10] = goal_root_vel
            self._else_root_states[env_ids, self._ref_actor_idx, 10:13] = goal_root_ang_vel
            self._else_rigid_body_pos[env_ids, rs_idx: re_idx] = global_pos
            self._else_rigid_body_rot[env_ids, rs_idx: re_idx] = global_rot
            self._else_rigid_body_vel[env_ids, rs_idx: re_idx] = global_vel
            self._else_rigid_body_ang_vel[env_ids, rs_idx: re_idx] = global_ang_vel

            self._else_dof_pos[
                env_ids, ds_idx: de_idx
            ] = goal_dof_pos
            self._else_dof_vel[
                env_ids, ds_idx: de_idx
            ] = goal_dof_vel

        if not self.reset_call_flag:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._root_states),
                gymtorch.unwrap_tensor(self.ref_actor_indices),
                len(self.ref_actor_indices),
            )

            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self._dof_state),
                gymtorch.unwrap_tensor(self.ref_dof_actor_indices),
                len(self.ref_dof_actor_indices),
            )

    def eval_tracking_metric(self,):
        assert not self._enable_early_termination
        if not hasattr(self, "sim_rigid_body_pos"):
            self.sim_rigid_body_pos = torch.zeros_like(self._ref_rigid_body_pos_buf)
            # initialize first one
            self.sim_rigid_body_pos[self.all_env_ids, 0] = self._ref_rigid_body_pos_buf[self.all_env_ids, 0]
        # update
        self.sim_rigid_body_pos[self.all_env_ids, self.progress_buf] = self._humanoid_rigid_body_pos

        # last
        if self.progress_buf[0] == self.max_episode_length - 1:
            sim_rigid_body_pos = self.sim_rigid_body_pos[:, :self.max_episode_length]
            goal_rigid_body_pos = self._ref_rigid_body_pos_buf[:, :self.max_episode_length]
            sim_rigid_body_pos = sim_rigid_body_pos.detach().cpu().numpy()
            goal_rigid_body_pos = goal_rigid_body_pos.detach().cpu().numpy()
            self.metric = compute_metrics_lite(sim_rigid_body_pos, goal_rigid_body_pos)
            delattr(self, "sim_rigid_body_pos")
        return

    def store_sim_states(self, query=["root", "dof_pos", "dof_vel", "local_ee_pos", "local_ee_contacts", "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel"]):
        assert not self._enable_early_termination
        if not hasattr(self, "stored_states_dict"):
            self.stored_states_dict = dict()
            self.deviated = torch.zeros_like(self.reset_buf)

        # calculate validity
        temp_reset_buf = self.reset_buf.clone()
        temp_terminate_buf = self._terminate_buf.clone()
        self._compute_reset(early_termination=True)
        self.deviated = torch.logical_or(self.deviated, self._terminate_buf)
        self.reset_buf[:] = temp_reset_buf
        self._terminate_buf[:] = temp_terminate_buf

        # Root states
        if "root" in query:
            if not hasattr(self, "sim_root_states"):
                self.sim_root_states = torch.zeros_like(self._ref_root_states_buf)
                self.stored_states_dict["root"] = self.sim_root_states
            self.sim_root_states[self.all_env_ids, self.progress_buf - 1] = self._humanoid_root_states
            query.pop(query.index("root"))

        # Dof positions
        if "dof_pos" in query:
            if not hasattr(self, "sim_dof_pos"):
                self.sim_dof_pos = torch.zeros_like(self._ref_dof_pos_buf)
                self.stored_states_dict["dof_pos"] = self.sim_dof_pos
            self.sim_dof_pos[self.all_env_ids, self.progress_buf - 1] = self._humanoid_dof_pos
            query.pop(query.index("dof_pos"))

        # Dof velocities
        if "dof_vel" in query:
            if not hasattr(self, "sim_dof_vel"):
                self.sim_dof_vel = torch.zeros_like(self._ref_dof_vel_buf)
                self.stored_states_dict["dof_vel"] = self.sim_dof_vel
            self.sim_dof_vel[self.all_env_ids, self.progress_buf - 1] = self._humanoid_dof_vel
            query.pop(query.index("dof_vel"))

        # rigid_body_pos
        if "rigid_body_pos" in query:
            if not hasattr(self, "sim_rigid_body_pos"):
                self.sim_rigid_body_pos = torch.zeros_like(self._ref_rigid_body_pos_buf)
                self.stored_states_dict["rigid_body_pos"] = self.sim_rigid_body_pos
            self.sim_rigid_body_pos[self.all_env_ids, self.progress_buf - 1] = self._humanoid_rigid_body_pos
            query.pop(query.index("rigid_body_pos"))

        # rigid_body_rot
        if "rigid_body_rot" in query:
            if not hasattr(self, "sim_rigid_body_rot"):
                self.sim_rigid_body_rot = torch.zeros_like(self._ref_rigid_body_rot_buf)
                self.stored_states_dict["rigid_body_rot"] = self.sim_rigid_body_rot
            self.sim_rigid_body_rot[self.all_env_ids, self.progress_buf - 1] = self._humanoid_rigid_body_rot
            query.pop(query.index("rigid_body_rot"))

        # rigid_body_vel
        if "rigid_body_vel" in query:
            if not hasattr(self, "sim_rigid_body_vel"):
                self.sim_rigid_body_vel = torch.zeros_like(self._ref_rigid_body_vel_buf)
                self.stored_states_dict["rigid_body_vel"] = self.sim_rigid_body_vel
            self.sim_rigid_body_vel[self.all_env_ids, self.progress_buf - 1] = self._humanoid_rigid_body_vel
            query.pop(query.index("rigid_body_vel"))

        # rigid_body_ang_vel
        if "rigid_body_ang_vel" in query:
            if not hasattr(self, "sim_rigid_body_ang_vel"):
                self.sim_rigid_body_ang_vel = torch.zeros_like(self._ref_rigid_body_ang_vel_buf)
                self.stored_states_dict["rigid_body_ang_vel"] = self.sim_rigid_body_ang_vel
            self.sim_rigid_body_ang_vel[self.all_env_ids, self.progress_buf - 1] = self._humanoid_rigid_body_ang_vel
            query.pop(query.index("rigid_body_ang_vel"))

        # Root-relative End-effector positions (in root heading coordinate)
        if "local_ee_pos" in query:
            if not hasattr(self, "sim_local_ee_pos"):
                self.sim_local_ee_pos = torch.zeros_like(self._ref_rigid_body_pos_buf[:, :, self._key_body_ids]) # (B, T, N, 3)
                self.stored_states_dict["local_ee_pos"] = self.sim_local_ee_pos
            # normalize positions in root heading coordinate
            root_pos = self._humanoid_root_states[:, :3]
            root_rot = self._humanoid_root_states[:, 3:7]
            heading_rot = calc_heading_quat_inv(root_rot)

            root_pos_expand = root_pos.unsqueeze(-2)
            key_body_pos = self._humanoid_rigid_body_pos[:, self._key_body_ids] 
            key_body_pos_shape = key_body_pos.shape
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
            local_end_pos = local_end_pos.reshape(*key_body_pos_shape)
            self.sim_local_ee_pos[self.all_env_ids, self.progress_buf - 1] = local_end_pos
            query.pop(query.index("local_ee_pos"))

        # Local contact forces (in root heading coordinate)
        if "local_ee_contacts" in query:
            if not hasattr(self, "sim_local_ee_contacts"):
                self.sim_local_ee_contacts = torch.zeros_like(self._ref_rigid_body_pos_buf[:, :, self._key_body_ids]) # (B, T, N, 3)
                self.stored_states_dict["local_ee_contacts"] = self.sim_local_ee_contacts
            # normalize forces in root heading coordinate
            root_rot = self._humanoid_root_states[:, 3:7]
            heading_rot = calc_heading_quat_inv(root_rot)

            root_pos_expand = root_pos.unsqueeze(-2)
            key_body_contacts = self._humanoid_contact_forces[:, self._key_body_ids] 
            key_body_contacts_shape = key_body_contacts.shape

            heading_rot_expand = heading_rot.unsqueeze(-2)
            heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
            flat_end_contacts = key_body_contacts.view(
                key_body_contacts.shape[0] * key_body_contacts.shape[1],
                key_body_contacts.shape[2],
            )
            flat_heading_rot = heading_rot_expand.view(
                heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
                heading_rot_expand.shape[2],
            )
            local_end_contacts = my_quat_rotate(flat_heading_rot, flat_end_contacts)
            local_end_contacts = local_end_contacts.reshape(*key_body_contacts_shape)
            self.sim_local_ee_contacts[self.all_env_ids, self.progress_buf - 1] = local_end_contacts
            query.pop(query.index("local_ee_contacts"))
        if len(query) != 0:
            raise NotImplementedError("Unidentified query for states storing.")

        # last
        if self.progress_buf[0] == self.max_episode_length - 1:
            def tuple_to_str(tup):
                ret = "("
                for v in tup:
                    ret += str(v) + ","
                ret += ")"
                return ret

            # lazy import
            import datetime, pwd
            now = datetime.datetime.now()
            datestr = now.strftime("%Y-%m-%d")
            valid_ids = torch.where(torch.logical_not(self.deviated))[0]
            user_name = pwd.getpwuid(os.getuid())[0]
            dataset_name = self.cfg["env"]["motion_file"]
            checkpoint_name = self.cfg["env"]["load_path"].split('/')[-1].split('.')[0]

            data_root_dir = os.path.join(os.path.curdir, 'trajectories', checkpoint_name, datestr)
            datestr_hms = now.strftime("%Y-%m-%d:%H-%M-%S")
            key_body_str = tuple_to_str(KEY_BODY_NAMES)
            self._contact_bodies
            os.makedirs(data_root_dir, exist_ok=True)
            infos = [
                "Generated by %s" % user_name,
                "Generated at %s" % datestr_hms,
                "Dataset : %s" % dataset_name,
                "Checkpoint : %s" % checkpoint_name,
                "", 
                "Total size (in seconds) : %.1fs" % (valid_ids.shape[0] * (self.max_episode_length - 1) / 30),
                "Number of Episodes : %d" % valid_ids.shape[0], 
                "Number of Frames per Episode : %d" % (self.max_episode_length - 1), 
                "", 
                "End-effector Lists : %s" % key_body_str,
                "", 
                "Shape of arrays : "
            ]

            for key, tensor in self.stored_states_dict.items():
                save_tensor = tensor[valid_ids, :self.max_episode_length - 1]
                save_array = save_tensor.detach().cpu().numpy()
                np.save(os.path.join(data_root_dir, f'{key}.npy'), save_array)
                infos.append(f"  {key} : {tuple_to_str(save_array.shape)}")
                if key == "root":
                    saved_root_rot = save_tensor[..., 3:7]
                if key == "dof_pos":
                    B, T, _ = save_tensor.shape
                    saved_dof_pos = save_tensor
                    local_rotation_tensor = dof_to_rotation(saved_root_rot.view(-1, 4), saved_dof_pos.view(-1, 28)).view(B, T, 15, 4)
                    save_array = local_rotation_tensor.detach().cpu().numpy()
                    np.save(os.path.join(data_root_dir, f'local_rotation.npy'), save_array)
                    infos.append(f"  local_rotation : {tuple_to_str(save_array.shape)}")

            
            with open(os.path.join(data_root_dir, 'info.txt'), 'w') as f:
                for line in infos:
                    f.write(f"{line}\n")
            print("Successfully saved at %s" % data_root_dir)
            exit(1)
        return
#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_ref_diff_observations(
    ref_rigid_body_pos,
    ref_rigid_body_rot,
    ref_rigid_body_vel,
    ref_rigid_body_ang_vel,
    sim_rigid_body_pos,
    sim_rigid_body_rot,
    sim_rigid_body_vel,
    sim_rigid_body_ang_vel,
    sim_root_rot,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
    N, J, _ = ref_rigid_body_pos.shape
    # heading rots
    sim_heading_rot = calc_heading_quat_inv(sim_root_rot)
    sim_heading_rot_expand = sim_heading_rot[:, None].repeat(1, J, 1)
    sim_heading_rot_flat = sim_heading_rot_expand.reshape(N * J, -1)

    # rigid body pos
    ref_rigid_body_pos_flat = ref_rigid_body_pos.reshape(N * J, -1)
    sim_rigid_body_pos_flat = sim_rigid_body_pos.reshape(N * J, -1)
    ref_rigid_body_pos_obs = my_quat_rotate(
        sim_heading_rot_flat, ref_rigid_body_pos_flat - sim_rigid_body_pos_flat
    ).view(N, -1)

    # rigid body rot
    ref_rigid_body_rot_flat = ref_rigid_body_rot.reshape(N * J, -1)
    sim_rigid_body_rot_flat = sim_rigid_body_rot.reshape(N * J, -1)
    ref_rigid_body_rot_obs = quat_mul(ref_rigid_body_rot_flat, quat_conjugate(sim_rigid_body_rot_flat))
    ref_rigid_body_rot_obs = quat_to_rotation_6d(ref_rigid_body_rot_obs).view(N, -1)

    # rigid body vel
    ref_rigid_body_vel_flat = ref_rigid_body_vel.reshape(N * J, -1)
    sim_rigid_body_vel_flat = sim_rigid_body_vel.reshape(N * J, -1)
    ref_rigid_body_vel_obs = my_quat_rotate(
        sim_heading_rot_flat, ref_rigid_body_vel_flat - sim_rigid_body_vel_flat
    ).view(N, -1)

    # rigid body ang vel
    ref_rigid_body_ang_vel_obs = (ref_rigid_body_ang_vel - sim_rigid_body_ang_vel).view(N, -1)

    obs = torch.cat(
        (
            ref_rigid_body_pos_obs,
            ref_rigid_body_rot_obs,
            ref_rigid_body_vel_obs,
            ref_rigid_body_ang_vel_obs,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
# inspired by reference - deepmimic (Peng, 2018)
def compute_deepmimic_reward(
    curr_dof_pos,
    goal_dof_pos,
    curr_dof_vel,
    goal_dof_vel,
    curr_ee_pos,
    goal_ee_pos,
    curr_root_pos,
    goal_root_pos,
    multiplication,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    """
    obs = {
        root_states (13),
        dof_pos (28),
        dof_vel (28),
        key_pos (12)
    }
    """
    # compute errors
    dof_pos_error = (curr_dof_pos - goal_dof_pos).pow(2).mean(dim=-1)

    dof_vel_error = (curr_dof_vel - goal_dof_vel).pow(2).mean(dim=-1)

    ee_pos_error = (curr_ee_pos - goal_ee_pos).pow(2).sum(dim=-1).mean(dim=-1)

    root_pos_error = (curr_root_pos - goal_root_pos).pow(2).sum(dim=-1)

    # compute reward
    body_rot_reward = torch.exp(-2 * dof_pos_error)
    body_ang_vel_reward = torch.exp(-0.1 * dof_vel_error)
    ee_pos_reward = torch.exp(-40 * ee_pos_error)
    root_pos_reward = torch.exp(-10 * root_pos_error)

    # deepmimic version
    if multiplication:
        reward = body_rot_reward * body_ang_vel_reward * ee_pos_reward * root_pos_reward
    else:
        reward = 0.65 * body_rot_reward + 0.1 * body_ang_vel_reward + 0.15 * ee_pos_reward + 0.1 * root_pos_reward
    # print(body_rot_reward[0].item(), body_ang_vel_reward[0].item(), ee_pos_reward[0].item(), root_pos_reward[0].item())

    return reward


@torch.jit.script
# inspired by reference - PHC (Luo et al., 2023)
def compute_phc_reward(
    curr_rigid_body_pos,
    goal_rigid_body_pos,
    curr_rigid_body_rot,
    goal_rigid_body_rot,
    curr_rigid_body_vel,
    goal_rigid_body_vel,
    curr_rigid_body_ang_vel,
    goal_rigid_body_ang_vel,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor

    # compute errors
    N, J, _ = curr_rigid_body_rot.shape

    body_pos_error = (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=(-1, -2))

    curr_rigid_body_rot = curr_rigid_body_rot.reshape(-1, 4)
    goal_rigid_body_rot = goal_rigid_body_rot.reshape(-1, 4)
    body_rot_error = quat_diff_rad(curr_rigid_body_rot, goal_rigid_body_rot).view(N, J).pow(2).mean(dim=-1)

    body_vel_error = (curr_rigid_body_vel - goal_rigid_body_vel).pow(2).sum(dim=-1).mean(dim=-1)

    body_ang_vel_error = (curr_rigid_body_ang_vel - goal_rigid_body_ang_vel).pow(2).sum(dim=-1).mean(dim=-1)

    # compute reward
    body_pos_reward = torch.exp(-100 * body_pos_error)
    body_rot_reward = torch.exp(-10 * body_rot_error)
    body_vel_reward = torch.exp(-0.1 * body_vel_error)
    body_ang_vel_reward = torch.exp(-0.1 * body_ang_vel_error)

    # deepmimic version
    reward = 0.5 * body_pos_reward + 0.3 * body_rot_reward + 0.1 * body_vel_reward + 0.1 * body_ang_vel_reward
    # print(body_pos_reward[0].item(), body_rot_reward[0].item(), body_vel_reward[0].item(), body_ang_vel_reward[0].item())

    return reward

@torch.jit.script
def compute_energy_reward(
    actions,
    dof_vel,
    coef
):
    # type: (Tensor, Tensor, float) -> Tensor
    mult = (actions * dof_vel).pow(2).sum(dim=-1)
    energy_rew = torch.exp(-coef * mult)
    return energy_rew


@torch.jit.script
def compute_imitation_reset_mean(
    reset_buf,
    progress_buf,
    curr_rigid_body_pos,
    goal_rigid_body_pos,
    contact_body_ids,
    max_episode_length,
    enable_early_termination,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    # inspired by "Perpetual Humanoid Control for Real-time Simulated Avatars" (2023)
    terminated = torch.zeros_like(reset_buf)

    # rigid_body_pos -> B, K, 3
    if enable_early_termination:
        pos_dist = (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).sqrt()
        pos_dist[:, contact_body_ids] = 0
        mean_pos_dist = pos_dist.sum(dim=-1) / (pos_dist.shape[-1] - contact_body_ids.shape[0])
        has_deviated = mean_pos_dist > 0.5
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_imitation_reset_root_dist(
    reset_buf,
    progress_buf,
    curr_root_pos,
    goal_root_pos,
    max_episode_length,
    enable_early_termination,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    # inspired by "Perpetual Humanoid Control for Real-time Simulated Avatars" (2023)
    terminated = torch.zeros_like(reset_buf)

    # rigid_body_pos -> B, K, 3
    if enable_early_termination:
        root_dist = (curr_root_pos - goal_root_pos).pow(2).sum(dim=-1).sqrt()
        has_deviated = root_dist > 1.0
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

@torch.jit.script
def compute_imitation_reset_max(
    reset_buf,
    progress_buf,
    curr_rigid_body_pos,
    goal_rigid_body_pos,
    contact_body_ids,
    max_episode_length,
    enable_early_termination,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor]
    # inspired by "Perpetual Humanoid Control for Real-time Simulated Avatars" (2023)
    terminated = torch.zeros_like(reset_buf)

    # rigid_body_pos -> B, K, 3
    if enable_early_termination:
        pos_dist = (curr_rigid_body_pos - goal_rigid_body_pos).pow(2).sum(dim=-1).sqrt()
        pos_dist[:, contact_body_ids] = 0
        max_pos_dist = torch.max(pos_dist, dim=-1).values
        has_deviated = max_pos_dist > 0.5
        terminated = torch.where(has_deviated, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

"""Translate dof pos to local rotation for each joint"""
@torch.jit.script
def dof_to_rotation(root_rot, dof_pos):
    # type: (Tensor, Tensor) -> Tensor
    body_nums = 15
    dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    dof_pos_shape = dof_pos.shape
    local_rotation_shape = dof_pos_shape[:-1] + (body_nums, 4)
    local_rotation = torch.zeros(local_rotation_shape, device=dof_pos.device)
    local_rotation[:, 0] = root_rot
    unit_quat = torch.zeros_like(root_rot)
    unit_quat[..., -1] = 1 # (x, y, z, w)
    offset_count = 0
    for j in range(1, body_nums):
        if j not in dof_body_ids:
            local_rotation[:, j] = unit_quat
            continue
        joint_offset = dof_offsets[offset_count]
        joint_size = dof_offsets[offset_count + 1] - joint_offset
        offset_count += 1
        joint_pos = dof_pos[..., joint_offset:joint_offset + joint_size]
        if joint_size == 3:
            joint_quat = exp_map_to_quat(joint_pos)
        elif joint_size == 1:
            heading = joint_pos[..., 0]
            axis = torch.zeros_like(unit_quat[..., :3])
            axis[..., 1] = 1
            joint_quat = quat_from_angle_axis(heading, axis)
        else:
            print("Unsupported joint type")
            assert False
        local_rotation[:, j] = joint_quat
    return local_rotation
