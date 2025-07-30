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
    dof_to_obs,
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
    quat_to_angle_axis
)

PLANE_CONTACT_THRESHOLD = 0.05  # at the initial frame, foot must higher than this value
# Obses
# NUM_OBS - Proprioceptive states (defined in humanoid_amp_base.py)
NUM_NAV_OBS = 2 # target location : (relative position in the plane) / target heading : (goal heading direction relative to root)

# Target Location
RESPAWN_FREQUENCY = 120 # (frame)
MAX_RESPAWN_DISTANCE = 10
MIN_RESPAWN_DISTANCE = 5
MIN_TARGET_SPEED = 1

BODY_COLORS = [
    [114, 125, 140], # blue
    [140, 114, 114], # red
    [114, 140, 125], # green
    [140, 114, 139], # purple
    [140, 125, 114], # orange
]
UPPER_LOWER_BODY_GROUP = {}
for j in range(15):
    UPPER_LOWER_BODY_GROUP[j] = 0 if j in [k for k in range(9)] else 1

TRUNK_LIMBS_BODY_GROUP = {}
for j in range(15):
    if j in [k for k in range(0, 3)]:
        TRUNK_LIMBS_BODY_GROUP[j] = 0 # trunk
    elif j in [k for k in range(3, 6)]:
        TRUNK_LIMBS_BODY_GROUP[j] = 1 # right arm
    elif j in [k for k in range(6, 9)]:
        TRUNK_LIMBS_BODY_GROUP[j] = 2 # left arm
    elif j in [k for k in range(9, 12)]:
        TRUNK_LIMBS_BODY_GROUP[j] = 3 # right leg
    elif j in [k for k in range(12, 15)]:
        TRUNK_LIMBS_BODY_GROUP[j] = 4 # left leg

DOF_TO_BODY_IDS = [
    1, # 0 - torso
    1, # 1 - torso
    1, # 2 - torso
    2, # 3 - head
    2, # 4 - head
    2, # 5 - head
    3, # 6 - right_upper_arm
    3, # 7 - right_upper_arm
    3, # 8 - right_upper_arm
    4, # 9 - right lower arm
    6, # 10 - left upper arm
    6, # 11 - left upper arm
    6, # 12 - left upper arm
    7, # 13 - left lower arm
    9, # 14 - right thigh
    9, # 15 - right thigh
    9, # 16 - right thigh
    10, # 17 - right shin
    11, # 18 - right foot
    11, # 19 - right foot
    11, # 20 - right foot
    12, # 21 - left thigh
    12, # 22 - left thigh
    12, # 23 - left thigh
    13, # 24 - left shin
    14, # 25 - left foot
    14, # 26 - left foot
    14, # 27 - left foot
]

class HumanoidNavigation(HumanoidAMPBase):
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

        # render
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidNavigation.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        # goal
        self._num_next_obs_steps = cfg["env"].get("numNextObsSteps", 1)
        self.goal_obs_space = spaces.Box(np.ones(self.num_goal_obs) * -np.Inf, np.ones(self.num_goal_obs) * np.Inf)
        self.clip_goal_obs = self.cfg["env"].get("clipGoalObservations", np.Inf)

        # motion sampling related
        fps = round(1 / (self.cfg["sim"]["dt"] + 1e-7))  # assume fps is always integer

        # render
        self.cfg["env"]["renderFPS"] = 60
        if self.cfg["env"]["kinematic"]:
            self.ori_control_freq_inv = self.cfg["env"]["controlFrequencyInv"]
            self.cfg["env"]["controlFrequencyInv"] = 1
            self.cfg["env"]["renderFPS"] = 30
            self._enable_early_termination = False
        
        self.render_every = max(fps // self.cfg["env"]["renderFPS"], 1)

        self._display_goal = self.cfg["env"]["test"]

        self.eval_jitter = cfg["env"].get("eval_jitter", False) if cfg["env"]["test"] else False

        # Mode
        self.target_location = cfg["env"]["target_location"]
        if not self.target_location:
            self.min_target_heading_speed = cfg["env"]["heading_speed"]

        # Inference
        self._random_effort_cut = self.cfg["env"].get("random_effort_cut", False)

        # capture default pose
        if self.cfg["env"]["capture_default_pose"]:
            self.cfg["env"]["episodeLength"] = 2
            self.cfg["env"]["target_episodes"] = [0]
            self.cfg["env"]["motion_file"] = "LaFAN1/train/walk1_subject1.npy"

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )
        if self.target_location:
            # goal related buffer
            self.goal_ground_pos = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
        else:
            # goal related buffer
            self.goal_ground_dir = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=self.device)
            # rendering
            if self.cfg["env"]["test"]:
                self.all_env_ids = torch.arange(self.num_envs, device=self.device)

        # motion loading
        motion_file = cfg["env"].get("motion_file", "amp_humanoid_backflip.npy")
        motion_file_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/motions/")

        if not isinstance(motion_file, omegaconf.listconfig.ListConfig):
            motion_file_list = [motion_file,]
        else:
            motion_file_list = motion_file

        motion_file_path = []
        for motion_file in motion_file_list:
            if isinstance(motion_file, str):
                motion_file = os.path.join(motion_file_root_path, motion_file)
                # Case 1 : if it is single motion file
                if motion_file.split(".")[-1] == "npy":
                    motion_file_path.append(motion_file)
                # Case 2 : if it is directory
                elif os.path.isdir(motion_file):
                    temp_motion_file_path = list(Path(motion_file).rglob("*.npy"))
                    motion_file_path.extend(temp_motion_file_path)
                else:
                    raise NotImplementedError()
            else:
                raise NotImplementedError()
        self.motion_file_name = [str(m).split('/')[-1].split('.')[0] for m in motion_file_path]
        self._load_motion(motion_file_path)

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
        if self._display_goal:
            if self.target_location:
                self.green_actor_indices = self.global_actor_indices[:, 2].clone()
            else:
                self.arrow_actor_indices = self.global_actor_indices[:, 1:3].flatten().clone()
        return

    def _prepare_else_assets(self):
        # NOTE THAT HUMANOID has priority
        # Ex) MUST DEFINE HUMANOID HANDLE AND THAN OBJECTS
        else_assets, else_start_poses, else_num_bodies, else_num_shapes = [], [], [], []
        self.num_else_actor = 0

        if self._display_goal:
            # goal
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
            if self.target_location:
                goal_asset_file = "urdf/hollow_cylinder/hollow_cylinder.urdf"
                goal_asset_options = gymapi.AssetOptions()
                goal_asset_options.fix_base_link = True
                goal_asset_options.disable_gravity = True
                goal_asset = self.gym.load_asset(self.sim, asset_root, goal_asset_file, goal_asset_options)

                # red goal - not arrived
                red_goal_pose = gymapi.Transform()
                red_goal_pose.p = gymapi.Vec3(0.0, 0, 0.0)
                self.red_actor_scale = 0.6
                red_goal_pose.p.z = -(self.red_actor_scale - 0.012)

                # green goal - arrived
                green_goal_pose = gymapi.Transform()
                green_goal_pose.p = gymapi.Vec3(0.0, 0, 0.0)
                self.green_actor_scale = 0.6001
                green_goal_pose.p.z = -100

                # append red and green
                else_assets.append(goal_asset)
                else_start_poses.append(red_goal_pose)
                else_num_bodies.append(self.gym.get_asset_rigid_body_count(goal_asset))
                else_num_shapes.append(self.gym.get_asset_rigid_shape_count(goal_asset))
                self.num_else_actor += 1

                else_assets.append(goal_asset)
                else_start_poses.append(green_goal_pose)
                else_num_bodies.append(self.gym.get_asset_rigid_body_count(goal_asset))
                else_num_shapes.append(self.gym.get_asset_rigid_shape_count(goal_asset))
                self.num_else_actor += 1
            else:
                arrow_asset_file = "urdf/arrow/arrow.urdf"
                arrow_asset_options = gymapi.AssetOptions()
                arrow_asset_options.disable_gravity = True
                arrow_asset = self.gym.load_asset(self.sim, asset_root, arrow_asset_file, arrow_asset_options)
                arrow_pose = gymapi.Transform()
                arrow_pose.p = gymapi.Vec3(0.0, 0, 2.0)

                # goal indicator
                else_assets.append(arrow_asset)
                else_start_poses.append(arrow_pose)
                else_num_bodies.append(self.gym.get_asset_rigid_body_count(arrow_asset))
                else_num_shapes.append(self.gym.get_asset_rigid_shape_count(arrow_asset))
                self.num_else_actor += 1

                # current indicator
                else_assets.append(arrow_asset)
                else_start_poses.append(arrow_pose)
                else_num_bodies.append(self.gym.get_asset_rigid_body_count(arrow_asset))
                else_num_shapes.append(self.gym.get_asset_rigid_shape_count(arrow_asset))
                self.num_else_actor += 1

                # light
                l_direction = gymapi.Vec3(0.0, -0.5, 1.0)
                l_color = gymapi.Vec3(1.0, 1.0, 1.0)
                l_ambient = gymapi.Vec3(0.1, 0.1, 0.1)
                self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)

        return else_assets, else_start_poses, else_num_bodies, else_num_shapes

    def _create_else_actors(self, env_ptr, env_idx, else_assets, else_start_poses):
        else_handles = []

        if self._display_goal:
            if self.target_location:
                contact_filter = (
                    1  # >1 : ignore all the collision, 0 : enable all collision, -1 : collision defined by robot file
                )

                # red goal first
                segmentation_id = 1 # segmentation ID used in segmentation camera sensors
                red_handle = self.gym.create_actor(
                    env_ptr,
                    else_assets[0],
                    else_start_poses[0],
                    "red_goal",
                    self.num_envs + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                goal_color = gymapi.Vec3(1.0, 0.5, 0.5)
                self.gym.set_actor_scale(env_ptr, red_handle, self.red_actor_scale) 
                self.gym.set_rigid_body_color(env_ptr, red_handle, 0, gymapi.MESH_VISUAL, goal_color)
                else_handles.append(red_handle)

                # green goal next
                segmentation_id = 1 # segmentation ID used in segmentation camera sensors
                green_handle = self.gym.create_actor(
                    env_ptr,
                    else_assets[1],
                    else_start_poses[1],
                    "green_goal",
                    self.num_envs * 2 + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                goal_color = gymapi.Vec3(0.5, 1.0, 0.5)
                self.gym.set_actor_scale(env_ptr, green_handle, self.green_actor_scale) 
                self.gym.set_rigid_body_color(env_ptr, green_handle, 0, gymapi.MESH_VISUAL, goal_color)
                else_handles.append(green_handle)
            else:
                contact_filter = (
                    1  # >1 : ignore all the collision, 0 : enable all collision, -1 : collision defined by robot file
                )

                # red goal first
                segmentation_id = 1 # segmentation ID used in segmentation camera sensors
                arrow_handle = self.gym.create_actor(
                    env_ptr,
                    else_assets[0],
                    else_start_poses[0],
                    "arrow",
                    self.num_envs + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                arrow_color = gymapi.Vec3(1.0, 0.5, 0.5)
                self.gym.set_rigid_body_color(env_ptr, arrow_handle, 0, gymapi.MESH_VISUAL, arrow_color)
                else_handles.append(arrow_handle)

                # indicator
                indicator_handle = self.gym.create_actor(
                    env_ptr,
                    else_assets[1],
                    else_start_poses[1],
                    "indicator",
                    self.num_envs * 2 + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                indicator_color = gymapi.Vec3(0.5, 1.0, 0.5)
                self.gym.set_rigid_body_color(env_ptr, indicator_handle, 0, gymapi.MESH_VISUAL, indicator_color)
                else_handles.append(indicator_handle)
        return else_handles

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        humanoid_asset, humanoid_start_pose, humanoid_num_bodies, humanoid_num_shapes = self._prepare_humanoid_asset()
        else_assets, else_start_poses, else_num_bodies, else_num_shapes = self._prepare_else_assets()
        else_assets, else_start_poses, else_num_bodies, else_num_shapes = self._maybe_add_white_backgrounds(else_assets, else_start_poses, else_num_bodies, else_num_shapes)

        max_agg_bodies = humanoid_num_bodies + np.array(else_num_bodies).sum().astype(np.int32)
        max_agg_shapes = humanoid_num_shapes + np.array(else_num_shapes).sum().astype(np.int32)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        self.dof_stiffness = []
        self.dof_damping = []

        # initialize num_dof / num_bodies
        self.num_dof = 0
        self.num_bodies = 0
        self._else_num_dof_offsets = []
        self._else_num_bodies_offsets = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            # begin aggregate mode
            self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            contact_filter = 0 if not self.cfg["env"]["kinematic"] else 1  # disable self-collision if kinematic mode
            handle = self.gym.create_actor(
                env_ptr, humanoid_asset, humanoid_start_pose, "humanoid", i, contact_filter, 0
            )

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            if self.cfg["env"]["kinematic"]:
                humanoid_color = gymapi.Vec3(0.8, 0.8, 0.8)
            else:
                if self.cfg["env"].get("white_mode", False):
                    # humanoid_color = gymapi.Vec3(0.5706, 0.649, 0.7863)
                    # humanoid_color = gymapi.Vec3(88 / 255, 96 / 255, 107 / 255)
                    if self.cfg["env"].get("prior_rollout", False):
                        base_color = np.array([114 / 255, 125 / 255, 140 / 255])
                        color = np.random.rand(3) * 0.3 - 0.15
                        color = base_color + color
                        humanoid_color = gymapi.Vec3(color[0], color[1], color[2])
                    else:
                        humanoid_color = gymapi.Vec3(114 / 255, 125 / 255, 140 / 255)
                else:
                    humanoid_color = gymapi.Vec3(0.4706, 0.549, 0.6863)
            for j in range(self.humanoid_num_bodies):
                if self.cfg["env"]["body_separation_viz"] == "upper-lower":
                    color = np.array(BODY_COLORS[UPPER_LOWER_BODY_GROUP[j]], dtype=np.float32) / 255
                    humanoid_color = gymapi.Vec3(color[0], color[1], color[2])
                elif self.cfg["env"]["body_separation_viz"] == "trunk-limbs":
                    color = np.array(BODY_COLORS[TRUNK_LIMBS_BODY_GROUP[j]], dtype=np.float32) / 255
                    humanoid_color = gymapi.Vec3(color[0], color[1], color[2])
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    humanoid_color,
                )
            
            # for visualizing random_effort_cut
            if self._random_effort_cut:
                # choose one leg and cut maximum effort for 3 random DoF in the chosen leg
                rand_leg_id = np.random.choice([0, 1], 1)
                # choose one arm and cut maximum effort for 3 random DoF in the chosen arm
                rand_arm_id = np.random.choice([0, 1], 1)

                if rand_leg_id == 0: # right leg
                    rand_joint_leg_id = np.random.choice([i for i in range(14, 21)], 3).tolist()
                else:
                    rand_joint_leg_id = np.random.choice([i for i in range(21, 28)], 3).tolist()
                rand_joint_id = rand_joint_leg_id

                if rand_arm_id == 0: # right arm
                    rand_joint_arm_id = np.random.choice([i for i in range(6, 10)], 2).tolist()
                else:
                    rand_joint_arm_id = np.random.choice([i for i in range(10, 14)], 2).tolist()
                rand_joint_id += rand_joint_arm_id
                
                cut_rate = np.random.rand(len(rand_joint_id)) * 0.1

                if self.cfg["env"]["test"] and not self.headless:
                    default_effort_cut_color = np.array([240, 50, 50], dtype=np.float32) / 255
                    default_humanoid_color = np.array([114, 125, 140], dtype=np.float32) / 255

                    rand_body_ids = [DOF_TO_BODY_IDS[k] for k in rand_joint_id]
                    mean_cut_rate = [0 for _ in range(15)]
                    selected_count = [0 for _ in range(15)]
                    for idx, rand_body_id in enumerate(rand_body_ids):
                        selected_count[rand_body_id] += 1
                        mean_cut_rate[rand_body_id] += cut_rate[idx] 
                    mean_cut_rate = [mean_cut_rate[i] / selected_count[i] if selected_count[i] != 0 else 0 for i in range(15)]

                    for j, cut_rate in enumerate(mean_cut_rate):
                        if cut_rate == 0:
                            continue
                        effort_cut_color = default_effort_cut_color * ((cut_rate * 5) + 0.5) + default_humanoid_color * (0.5 - (cut_rate * 5))
                        self.gym.set_rigid_body_color(
                            env_ptr,
                            handle,
                            j,
                            gymapi.MESH_VISUAL,
                            gymapi.Vec3(effort_cut_color[0], effort_cut_color[1], effort_cut_color[2]),
                        )
                

            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)

            else_handles = self._create_else_actors(env_ptr, i, else_assets, else_start_poses)
            else_handles = self._maybe_create_white_backgrounds_actors(env_ptr, i, else_assets, else_start_poses, else_handles)
            for else_handle in else_handles:
                self.envs.append(else_handle)

            # end aggregate mode
            self.gym.end_aggregate(env_ptr)

            if self._pd_control:
                dof_prop = self.gym.get_asset_dof_properties(humanoid_asset)
                if not self.cfg["env"]["kinematic"]:
                    dof_prop["driveMode"] = gymapi.DOF_MODE_POS
                    if self._set_effort_limit:
                        dof_prop["effort"] = self.motor_efforts.detach().cpu().numpy()
                    if self._random_effort_cut:
                        dof_prop["effort"][rand_joint_id] = self.motor_efforts.detach().cpu().numpy()[rand_joint_id] * cut_rate
                else:
                    dof_prop["driveMode"] = gymapi.DOF_MODE_NONE
                    dof_prop["friction"][:] = 1000000
                    dof_prop["damping"][:] = 1000000
                    dof_prop["stiffness"][:] = 1000000

                self.gym.set_actor_dof_properties(env_ptr, handle, dof_prop)
            self.set_else_actors_dof_properties(env_ptr, else_handles, else_assets)

        # count total num_dof / num_bodies
        actor_list = [handle] + else_handles
        self.num_actors = len(actor_list)
        dof_actor_indices_per_env = []
        for idx, actor in enumerate(actor_list):
            actor_dof_count = self.gym.get_actor_dof_count(env_ptr, actor)
            actor_bodies_count = self.gym.get_actor_rigid_body_count(env_ptr, actor)
            self._else_num_dof_offsets.append(self.num_dof)
            self._else_num_bodies_offsets.append(self.num_bodies)
            self.num_dof += actor_dof_count
            self.num_bodies += actor_bodies_count
            if actor_dof_count > 0:
                dof_actor_indices_per_env.append(idx)
        self._else_num_dof_offsets.append(self.num_dof)
        self._else_num_bodies_offsets.append(self.num_bodies)
        global_actor_indices = torch.arange(self.num_envs * self.num_actors, dtype=torch.int32, device=self.device)
        self.global_actor_indices = global_actor_indices.view(self.num_envs, self.num_actors)
        self.global_dof_actor_indices = self.global_actor_indices[:, dof_actor_indices_per_env].clone()

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.humanoid_num_dof):
            if dof_prop["lower"][j] > dof_prop["upper"][j]:
                self.dof_limits_lower.append(dof_prop["upper"][j])
                self.dof_limits_upper.append(dof_prop["lower"][j])
            else:
                self.dof_limits_lower.append(dof_prop["lower"][j])
                self.dof_limits_upper.append(dof_prop["upper"][j])
            # stiffness and damping
            self.dof_stiffness.append(dof_prop["stiffness"][j])
            self.dof_damping.append(dof_prop["damping"][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)
        self.dof_stiffness = to_torch(self.dof_stiffness, device=self.device)
        self.dof_damping = to_torch(self.dof_damping, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_ptr, handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_ptr, handle)

        if self._pd_control:
            self._build_pd_action_offset_scale()

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
        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()

        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def _compute_nav_goal_observations(self, env_ids=None):
        if self.target_location:
            if env_ids is None:
                root_rot = self._humanoid_root_states[:, 3:7]
                root_ground_pos = self._humanoid_root_states[:, :2]
                goal_ground_pos = self.goal_ground_pos
            else:
                root_rot = self._humanoid_root_states[env_ids, 3:7]
                root_ground_pos = self._humanoid_root_states[env_ids, :2]
                goal_ground_pos = self.goal_ground_pos[env_ids]

            goal_diff_obs = compute_goal_diff_observations(
                root_rot=root_rot,
                root_ground_pos=root_ground_pos,
                goal_ground_pos=goal_ground_pos
            )
            return goal_diff_obs
        else:
            if env_ids is None:
                root_rot = self._humanoid_root_states[:, 3:7]
                goal_ground_dir = self.goal_ground_dir
            else:
                root_rot = self._humanoid_root_states[env_ids, 3:7]
                goal_ground_dir = self.goal_ground_dir[env_ids]

            goal_heading_obs = compute_goal_heading_observations(
                root_rot=root_rot,
                goal_ground_dir=goal_ground_dir,
            )
            return goal_heading_obs

    def _compute_observations(self, env_ids=None):
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
        goal_obs = self._compute_nav_goal_observations(env_ids)
        return goal_obs

    def _compute_reward(self, actions):
        if self.target_location:
            root_rot = self._humanoid_root_states[:, 3:7]
            root_vel = self._humanoid_root_states[:, 7:10]
            root_ground_pos = self._humanoid_root_states[:, :2]
            goal_ground_pos = self.goal_ground_pos
            self.rew_buf[:] = compute_target_location_reward(
                root_vel=root_vel,
                root_rot=root_rot,
                root_ground_pos=root_ground_pos,
                goal_ground_pos=goal_ground_pos,
                min_target_speed=MIN_TARGET_SPEED
            )
        else:
            root_rot = self._humanoid_root_states[:, 3:7]
            root_vel = self._humanoid_root_states[:, 7:10]
            goal_ground_dir = self.goal_ground_dir
            self.rew_buf[:] = compute_target_heading_reward(
                root_vel=root_vel,
                root_rot=root_rot,
                goal_ground_dir=goal_ground_dir,
                min_target_speed=self.min_target_heading_speed
            )
        return

    def _update_target_location_goal(self, update_ids):
        root_pos = self._humanoid_root_states[update_ids, :2]
        # goal position
        random_disp = torch.empty_like(self.goal_ground_pos[update_ids]).normal_(generator=self.env_rng)
        random_disp = random_disp / (random_disp.norm(2, dim=-1, keepdim=True) + 1e-6)
        scale = (MAX_RESPAWN_DISTANCE - MIN_RESPAWN_DISTANCE)
        offset = MIN_RESPAWN_DISTANCE
        random_dist = torch.rand(size=(len(update_ids), ), device=self.device) * scale + offset
        new_goal = root_pos[..., :2] + random_dist[:, None] * random_disp
        self.goal_ground_pos[update_ids] = new_goal

        # for rendering
        if self._display_goal:  # only for rendering
            # red goal
            self._else_root_states[update_ids, 0, :2] = new_goal
            # # green goal
            self._else_root_states[update_ids, 1, :2] = new_goal
            self._else_root_states[update_ids, 1, 2] = -100
            if not self.reset_call_flag:
                self.reset_call_flag = True
                global_actor_indices = self.global_actor_indices[update_ids].flatten()
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self._root_states),
                    gymtorch.unwrap_tensor(global_actor_indices),
                    len(global_actor_indices),
                )

        return

    def _update_target_heading_goal(self, update_ids):
        random_dir = torch.empty_like(self.goal_ground_dir[update_ids]).normal_(generator=self.env_rng)
        random_dir = random_dir / (random_dir.norm(2, dim=-1, keepdim=True) + 1e-6)
        self.goal_ground_dir[update_ids] = random_dir

        # for rendering
        if self._display_goal:
            # rotation
            theta = torch.arccos(self.goal_ground_dir[update_ids, 0])
            sign = (self.goal_ground_dir[update_ids, 1] >= 0).float() * 2 - 1
            theta = theta * sign
            self._else_root_states[update_ids, 0, 3:5] = 0
            self._else_root_states[update_ids, 0, 5] = torch.sin(theta / 2)
            self._else_root_states[update_ids, 0, 6] = torch.cos(theta / 2)
                    # no need to call reset
        return

    def _update_goal(self):
        # update goal
        update_ids = torch.where(self.progress_buf % RESPAWN_FREQUENCY == 0)[0]
        if update_ids.shape[0] > 0:
            if self.target_location:
                self._update_target_location_goal(update_ids)
            else: # target heading
                self._update_target_heading_goal(update_ids)
        
        if self._display_goal and self.target_location:
            root_pos = self._humanoid_root_states[:, :2]
            goal_pos = self.goal_ground_pos
            dist = (root_pos - goal_pos).pow(2).sum(dim=-1).sqrt()
            is_close = dist < 0.5
            arrived_ids = torch.where(is_close)[0]
            self._else_root_states[arrived_ids, 1, 2] = self._else_root_states[0, 0, 2].item()
            if not self.reset_call_flag:
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self._root_states),
                    gymtorch.unwrap_tensor(self.green_actor_indices),
                    len(self.green_actor_indices),
            )

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_navigation_reset(
            self.reset_buf,
            self.progress_buf,
            self._humanoid_rigid_body_pos[:, 2, -1],
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_height,
        )
        self._update_goal()
        return

    @property
    def num_goal_obs(self) -> int:
        return NUM_NAV_OBS

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
                if i % self.render_every == 0:
                    self._motion_sync()
                    self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        if self.force_render and (self._display_goal):
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

    def _load_motion(self, motion_file):
        if self.cfg["env"]["test"]: # to faithfully eval
            self.env_rng = torch.Generator(device=self.device)
            self.env_rng.manual_seed(self.cfg["env"]["seed"])
        else:
            self.env_rng = None
        self._motion_lib = MotionLib(
            motion_file=motion_file,
            dof_body_ids=DOF_BODY_IDS,
            dof_offsets=DOF_OFFSETS,
            key_body_ids=self._key_body_ids.cpu().numpy(),
            device=self.device,
            min_len=1,
            motion_matching=False,
            generator=self.env_rng
        )
        return

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if self.force_render and (self._display_goal):
            self.reset_call_flag = True
        return

    def _reset_actors(self, env_ids):
        if self._state_init == HumanoidNavigation.StateInit.Default:
            self._reset_default(env_ids)
        elif (
            self._state_init == HumanoidNavigation.StateInit.Start or self._state_init == HumanoidNavigation.StateInit.Random
        ):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HumanoidNavigation.StateInit.Hybrid:
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
        return self._motion_lib.get_motion_state(motion_ids, motion_times, output_global=True)

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)

        if self._state_init == HumanoidNavigation.StateInit.Random or self._state_init == HumanoidNavigation.StateInit.Hybrid:
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == HumanoidNavigation.StateInit.Start:
            motion_times = torch.zeros(num_envs)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        if self.eval_jitter:
            motion_ids[:] = 0
            motion_times[:] = 0
        elif self.cfg["env"]["capture_default_pose"]:
            motion_ids[:] = 0
            motion_times[:] = 0
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
        ) = self._sample_ref_states(
            motion_ids,
            motion_times,
        )

        # avoid contact for the first frame
        min_body_pos = global_pos[:, :, 2].min(dim=-1).values # (B,)
        body_offset = min_body_pos - PLANE_CONTACT_THRESHOLD
        body_offset = torch.minimum(body_offset, torch.zeros_like(body_offset))
        if self.cfg["env"]["capture_default_pose"]:
            body_offset -= 100
            root_rot[:, 0] = 0
            root_rot[:, 1] = 0
            root_rot[:, 2] = -0.7071068
            root_rot[:, 3] = 0.7071068
        root_pos[..., 2] -= body_offset
        global_pos[..., 2] -= body_offset[:, None]
        key_pos[..., 2] -= body_offset[:, None]

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
    ):
        if self.target_location:
            # goal position
            random_disp = torch.empty_like(self.goal_ground_pos[env_ids]).normal_(generator=self.env_rng)
            random_disp = random_disp / (random_disp.norm(2, dim=-1, keepdim=True) + 1e-6)
            scale = (MAX_RESPAWN_DISTANCE - MIN_RESPAWN_DISTANCE)
            offset = MIN_RESPAWN_DISTANCE
            random_dist = torch.rand(size=(len(env_ids), ), device=self.device) * scale + offset
            new_goal = root_pos[..., :2] + random_dist[:, None] * random_disp
            self.goal_ground_pos[env_ids] = new_goal

            # for rendering
            if self._display_goal:  # only for rendering
                # red goal
                self._else_root_states[env_ids, 0, :2] = new_goal
                # green goal
                self._else_root_states[env_ids, 1, :2] = new_goal
                self._else_root_states[env_ids, 1, 2] = -100
        else:
            random_dir = torch.empty_like(self.goal_ground_dir[env_ids]).normal_(generator=self.env_rng)
            random_dir = random_dir / (random_dir.norm(2, dim=-1, keepdim=True) + 1e-6)
            self.goal_ground_dir[env_ids] = random_dir

            # for rendering
            if self._display_goal:
                # arrow
                theta = torch.arccos(self.goal_ground_dir[env_ids, 0])
                sign = (self.goal_ground_dir[env_ids, 1] >= 0).float() * 2 - 1
                theta = theta * sign
                self._else_root_states[env_ids, 0, 3:5] = 0
                self._else_root_states[env_ids, 0, 5] = torch.sin(theta / 2)
                self._else_root_states[env_ids, 0, 6] = torch.cos(theta / 2)
                self._else_root_states[env_ids, 0, :2] = self._humanoid_root_states[env_ids, :2]
                self._else_root_states[env_ids, 0, 7:9] = self._humanoid_root_states[env_ids, 7:9]

                # indicator
                root_vel = self._humanoid_root_states[env_ids, 7:9]
                root_dir = root_vel / root_vel.norm(2, dim=-1, keepdim=True)
                theta = torch.arccos(root_dir[:, 0])
                sign = (root_dir[:, 1] >= 0).float() * 2 - 1
                theta = theta * sign
                self._else_root_states[env_ids, 1, 3:5] = 0
                self._else_root_states[env_ids, 1, 5] = torch.sin(theta / 2)
                self._else_root_states[env_ids, 1, 6] = torch.cos(theta / 2)
                self._else_root_states[env_ids, 1, :2] = self._humanoid_root_states[env_ids, :2]
                self._else_root_states[env_ids, 1, 7:9] = self._humanoid_root_states[env_ids, 7:9]
        return

    def _motion_sync(self):
        if self._display_goal and not self.target_location:
            # arrow
            self._else_root_states[:, 0, :2] = self._humanoid_root_states[:, :2]
            self._else_root_states[:, 0, 7:9] = self._humanoid_root_states[:, 7:9]
            # indicator
            root_vel = self._humanoid_root_states[:, 7:9]
            root_dir = root_vel / root_vel.norm(2, dim=-1, keepdim=True)
            theta = torch.arccos(root_dir[:, 0])
            sign = (root_dir[:, 1] >= 0).float() * 2 - 1
            theta = theta * sign
            self._else_root_states[:, 1, 3:5] = 0
            self._else_root_states[:, 1, 5] = torch.sin(theta / 2)
            self._else_root_states[:, 1, 6] = torch.cos(theta / 2)
            self._else_root_states[:, 1, :2] = self._humanoid_root_states[:, :2]
            self._else_root_states[:, 1, 7:9] = self._humanoid_root_states[:, 7:9]

            if not self.reset_call_flag:
                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self._root_states),
                    gymtorch.unwrap_tensor(self.arrow_actor_indices),
                    len(self.arrow_actor_indices),
                )

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
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel,
            rigid_body_pos=global_pos,
            rigid_body_rot=global_rot,
            rigid_body_vel=global_vel,
            rigid_body_ang_vel=global_ang_vel,
        )
        self._set_else_states(
            env_ids=env_ids,
            root_pos=root_pos,
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
        
        if not self.cfg["env"]["capture_default_pose"]:
            if self.target_location:
                cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 4.0, 3)
            else:
                cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 2.0, 4)
            cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        else:
            cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 2, 101.5)
            cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 101.0)

        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        if not self.cfg["env"]["capture_default_pose"]:
            new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        else:
            new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 101.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_goal_diff_observations(
    root_rot,
    root_ground_pos,
    goal_ground_pos,
):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    # heading rots
    heading_rot = calc_heading_quat_inv(root_rot)
    disp = goal_ground_pos - root_ground_pos
    disp_aug = torch.cat([disp, torch.zeros_like(disp[:, :1])], dim=-1) # set z as 0
    aligned_disp = my_quat_rotate(heading_rot, disp_aug)
    return aligned_disp[:, :2]

@torch.jit.script
def compute_goal_heading_observations(
    root_rot,
    goal_ground_dir,
):
    # type: (Tensor, Tensor) -> Tensor
    # heading rots
    heading_rot = calc_heading_quat_inv(root_rot)
    goal_dir = goal_ground_dir
    goal_dir_aug = torch.cat([goal_dir, torch.zeros_like(goal_dir[:, :1])], dim=-1) # set z as 0
    aligned_goal_dir = my_quat_rotate(heading_rot, goal_dir_aug)
    return aligned_goal_dir[:, :2]

@torch.jit.script
def compute_target_location_reward(
    root_vel,
    root_rot,
    root_ground_pos,
    goal_ground_pos,
    min_target_speed,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float) -> Tensor
    root_vel = root_vel[:, :2]
    root_goal_disp = goal_ground_pos - root_ground_pos
    root_goal_dist = root_goal_disp.pow(2).sum(dim=-1)

    root_heading_theta, root_heading_axis = quat_to_angle_axis(
        calc_heading_quat(root_rot)
    )
    root_heading_theta = root_heading_theta * root_heading_axis[:, -1]
    root_heading_dir = torch.stack([torch.cos(root_heading_theta), torch.sin(root_heading_theta)], dim=-1)
    scaled_humanoid_speed = torch.max(
        torch.zeros_like(root_heading_theta),
        (root_vel * root_heading_dir).sum(dim=-1),
    )
    scaled_humanoid_vel = scaled_humanoid_speed[:, None] * root_heading_dir
    root_projected_speed = (root_goal_disp * scaled_humanoid_vel).sum(dim=-1) / (root_goal_dist.sqrt() + 1e-5)

    loc_reward = torch.exp(-0.5 * root_goal_dist)
    vel_reward = torch.exp(
        -(
            torch.max(
                torch.zeros_like(root_projected_speed),
                min_target_speed - root_projected_speed,
            ).pow(2)
        )
    )
    # when sufficiently near, then velocity reward is 1
    is_near = torch.where(root_goal_dist.sqrt() < 0.5)[0]
    vel_reward[is_near] = 1

    rew = 0.7 * loc_reward + 0.3 * vel_reward
    return rew

@torch.jit.script
def compute_target_heading_reward(
    root_vel,
    root_rot,
    goal_ground_dir,
    min_target_speed,
):
    # type: (Tensor, Tensor, Tensor, float) -> Tensor
    root_vel = root_vel[:, :2]
    root_heading_theta, root_heading_axis = quat_to_angle_axis(
        calc_heading_quat(root_rot)
    )
    root_heading_theta = root_heading_theta * root_heading_axis[:, -1]
    root_heading_dir = torch.stack([torch.cos(root_heading_theta), torch.sin(root_heading_theta)], dim=-1)
    scaled_humanoid_speed = torch.max(
        torch.zeros_like(root_heading_theta),
        (root_vel * root_heading_dir).sum(dim=-1),
    )
    scaled_humanoid_vel = scaled_humanoid_speed[:, None] * root_heading_dir
    root_projected_speed = (goal_ground_dir * scaled_humanoid_vel).sum(dim=-1)

    vel_reward = torch.exp(
        -2.5 * (
            # torch.max(
            #     torch.zeros_like(root_projected_speed),
            #     min_target_speed - root_projected_speed,
            # ).pow(2)
            (min_target_speed - root_projected_speed).pow(2)
        )
    )

    norm_root_vel = root_vel / (root_vel.norm(2, dim=-1, keepdim=True) + 1e-6)
    dir_similarity = (norm_root_vel * goal_ground_dir).sum(dim=-1) # [-1, 1]
    heading_reward = torch.exp(-5 * (1 - dir_similarity).pow(2))
    return vel_reward * heading_reward

@torch.jit.script
def compute_navigation_reset(
    reset_buf,
    progress_buf,
    head_height,
    max_episode_length,
    enable_early_termination,
    termination_height,
):
    # type: (Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:

        has_fallen = head_height < termination_height

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated