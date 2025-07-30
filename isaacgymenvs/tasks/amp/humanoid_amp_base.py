
# Copyright (c) 2018-2023, NVIDIA Corporation
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
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import os

import numpy as np
import torch
from isaacgym import gymapi, gymtorch

from isaacgymenvs.utils.torch_jit_utils import (
    calc_heading_quat_inv,
    exp_map_to_quat,
    get_axis_params,
    my_quat_rotate,
    quat_mul,
    quat_to_rotation_6d,
    to_torch,
)

from ..base.vec_task import VecTask

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
# abdomen (0-3)/ neck (3-6)/ right shoulder (6-9)/ right elbow (9-10)/ left shoulder (10-13)/ left elbow (13-14)
# right hip (14-17)/ right knee (17-18)/ right ankle (18-21) / left hip (21-24) / left knee (24-25) / left ankle (25-28)
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
NUM_OBS = 13 + 52 + 28 + 12  # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
NUM_ACTIONS = 28


KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]
GRID_LINES_NUM = 50
GRID_LENGTH = 50
GRID_THICKNESS = 0.05
GRID_COLOR = 0.6

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

class HumanoidAMPBase(VecTask):
    def __init__(
        self,
        config,
        rl_device,
        sim_device,
        graphics_device_id,
        headless,
        virtual_screen_capture,
        force_render,
    ):
        self.cfg = config

        self._pd_control = self.cfg["env"]["pdControl"]
        self._set_effort_limit = self.cfg["env"].get("setEffortLimit", False)
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        # visualization - only matters when testing and rendering
        self.cfg["env"]["body_separation_viz"] = "total" if headless else self.cfg["env"]["body_separation_viz"]
        assert self.cfg["env"]["body_separation_viz"] in ["total", "upper-lower", "trunk-limbs"]

        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            headless=headless,
            virtual_screen_capture=virtual_screen_capture,
            force_render=force_render,
        )

        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # root states
        self._root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, -1, 13)
        self._humanoid_root_states = self._root_states[:, 0]
        self._else_root_states = self._root_states[:, 1:]

        self._initial_root_states = self._root_states.clone()
        self._initial_humanoid_root_states = self._initial_root_states[:, 0]
        self._initial_humanoid_root_states[:, 7:13] = 0
        self._initial_else_root_states = self._initial_root_states[:, 1:]

        # dof states
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self._humanoid_dof_pos = self._dof_pos[:, : self.humanoid_num_dof]
        self._humanoid_dof_vel = self._dof_vel[:, : self.humanoid_num_dof]
        self._else_dof_pos = self._dof_pos[:, self.humanoid_num_dof :]
        self._else_dof_vel = self._dof_vel[:, self.humanoid_num_dof :]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)
        self._initial_humanoid_dof_pos = self._initial_dof_pos[:, : self.humanoid_num_dof]
        right_shoulder_x_handle = self.gym.find_actor_dof_handle(
            self.envs[0], self.humanoid_handles[0], "right_shoulder_x"
        )
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(
            self.envs[0], self.humanoid_handles[0], "left_shoulder_x"
        )
        self._initial_humanoid_dof_pos[:, right_shoulder_x_handle] = 0.5 * np.pi
        self._initial_humanoid_dof_pos[:, left_shoulder_x_handle] = -0.5 * np.pi

        # rigid body states
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        self._rigid_body_pos = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(contact_force_tensor).view(self.num_envs, self.num_bodies, 3)

        self._humanoid_rigid_body_pos = self._rigid_body_pos[:, : self.humanoid_num_bodies]
        self._humanoid_rigid_body_rot = self._rigid_body_rot[:, : self.humanoid_num_bodies]
        self._humanoid_rigid_body_vel = self._rigid_body_vel[:, : self.humanoid_num_bodies]
        self._humanoid_rigid_body_ang_vel = self._rigid_body_ang_vel[:, : self.humanoid_num_bodies]
        self._humanoid_contact_forces = self._contact_forces[:, : self.humanoid_num_bodies]

        self._else_rigid_body_pos = self._rigid_body_pos[:, self.humanoid_num_bodies :]
        self._else_rigid_body_rot = self._rigid_body_rot[:, self.humanoid_num_bodies :]
        self._else_rigid_body_vel = self._rigid_body_vel[:, self.humanoid_num_bodies :]
        self._else_rigid_body_ang_vel = self._rigid_body_ang_vel[:, self.humanoid_num_bodies :]
        self._else_contact_forces = self._contact_forces[:, self.humanoid_num_bodies :]

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._target_actions = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=torch.float32)

        if self.viewer != None:
            self._init_camera()

        return

    def get_obs_size(self):
        return NUM_OBS

    def get_action_size(self):
        return NUM_ACTIONS

    def create_sim(self):
        self.up_axis_idx = 2  # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        # # below line is apparently bug
        # # bug : if call refresh rigid body state when reset, then rigid body goes back to the value from last episode, even if it is manually reset to other value
        # fix this by adding additional toggle not to call rigid body state update when reset
        self._refresh_sim_tensors(reset=True)
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.envs[i]
            handle = self.humanoid_handles[i]

            for j in range(self.humanoid_num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(col[0], col[1], col[2]),
                )

        return

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _prepare_humanoid_asset(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")
        asset_file = "mjcf/amp_humanoid.xml"

        if "asset" in self.cfg["env"]:
            # asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        if self.cfg["env"]["kinematic"]:
            asset_options.fix_base_link = True
            asset_options.disable_gravity = True

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.humanoid_num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.humanoid_num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor(
            [start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w],
            device=self.device,
        )

        # num_bodies, num_shapes
        humanoid_num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        humanoid_num_shapes = self.gym.get_asset_rigid_shape_count(humanoid_asset)

        # save for keyframes
        self.humanoid_asset_root = asset_root
        self.humanoid_asset_file = asset_file
        self.humanoid_asset_options = asset_options
        self.humanoid_start_pose = start_pose
        return humanoid_asset, start_pose, humanoid_num_bodies, humanoid_num_shapes

    def _prepare_else_assets(self):
        return [], [], [], []

    def _create_else_actors(self):
        return []

    def set_else_actors_dof_properties(self, env_ptr, else_handles, else_assets):
        return

    def _maybe_add_white_backgrounds(self, else_assets, else_start_poses, else_num_bodies, else_num_shapes):
        if self.cfg["env"]["test"] and self.cfg["env"].get("white_mode", False):
            BOX_X, BOX_Y, BOX_Z = 1000, 1000, 0.2
            box_dims = gymapi.Vec3(BOX_X, BOX_Y, BOX_Z)
            box_asset_options = gymapi.AssetOptions()
            box_asset_options.fix_base_link = True
            # box_asset_options.collapse_fixed_joints = True
            box_asset_options.disable_gravity = True
            box_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, box_asset_options)
            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(0, 0, -BOX_Z / 2 + 1e-2)
            else_assets.append(box_asset)
            else_start_poses.append(box_pose)
            else_num_bodies.append(1)
            else_num_shapes.append(1)
            self.num_else_actor += 1
            
            # back plate
            BOX_X, BOX_Y, BOX_Z = 1000, 0.2, 1000
            box_dims = gymapi.Vec3(BOX_X, BOX_Y, BOX_Z)
            box_asset_options = gymapi.AssetOptions()
            box_asset_options.fix_base_link = True
            # box_asset_options.collapse_fixed_joints = True
            box_asset_options.disable_gravity = True
            box_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, box_asset_options)
            box_pose = gymapi.Transform()
            box_pose.p = gymapi.Vec3(0, 100, 0)
            else_assets.append(box_asset)
            else_start_poses.append(box_pose)
            else_num_bodies.append(1)
            else_num_shapes.append(1)
            self.num_else_actor += 1

            if GRID_LINES_NUM > 0:
                # grid lines - horizontal
                BOX_X, BOX_Y, BOX_Z = GRID_LENGTH, GRID_THICKNESS, 0.025
                box_dims = gymapi.Vec3(BOX_X, BOX_Y, BOX_Z)
                box_asset_options = gymapi.AssetOptions()
                box_asset_options.fix_base_link = True
                # box_asset_options.collapse_fixed_joints = True
                box_asset_options.disable_gravity = True
                box_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, box_asset_options)
                else_assets.append(box_asset)
                for i in range(GRID_LINES_NUM):
                    box_pose = gymapi.Transform()
                    box_pose.p = gymapi.Vec3(0, i - GRID_LINES_NUM // 2, 0)
                    else_start_poses.append(box_pose)
                    else_num_bodies.append(1)
                    else_num_shapes.append(1)
                    self.num_else_actor += 1

                # grid lines - vertical
                BOX_X, BOX_Y, BOX_Z = GRID_THICKNESS, GRID_LENGTH, 0.025
                box_dims = gymapi.Vec3(BOX_X, BOX_Y, BOX_Z)
                box_asset_options = gymapi.AssetOptions()
                box_asset_options.fix_base_link = True
                # box_asset_options.collapse_fixed_joints = True
                box_asset_options.disable_gravity = True
                box_asset = self.gym.create_box(self.sim, box_dims.x, box_dims.y, box_dims.z, box_asset_options)
                else_assets.append(box_asset)
                for i in range(GRID_LINES_NUM):
                    box_pose = gymapi.Transform()
                    box_pose.p = gymapi.Vec3(i - GRID_LINES_NUM // 2, 0, 0)
                    else_start_poses.append(box_pose)
                    else_num_bodies.append(1)
                    else_num_shapes.append(1)
                    self.num_else_actor += 1
        return else_assets, else_start_poses, else_num_bodies, else_num_shapes

    def _maybe_create_white_backgrounds_actors(self, env_ptr, env_idx, else_assets, else_start_poses, else_handles):
        if self.cfg["env"]["test"] and self.cfg["env"].get("white_mode", False):
            ground_asset = else_assets[-2] if GRID_LINES_NUM == 0 else else_assets[-4]
            ground_start_pose = else_start_poses[-2 - GRID_LINES_NUM * 2]
            contact_filter = (
                1  # >1 : ignore all the collision, 0 : enable all collision, -1 : collision defined by robot file
            )
            # handle for ground_plate
            actor_idx = len(else_assets)
            segmentation_id = 2 # segmentation ID used in segmentation camera sensors
            handle = self.gym.create_actor(
                env_ptr,
                ground_asset,
                ground_start_pose,
                "white_ground",
                self.num_envs * (actor_idx - 1) + env_idx,
                contact_filter,
                segmentation_id,
            )
            ground_color = gymapi.Vec3(1, 1, 1)
            self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, ground_color)
            else_handles.append(handle)

            # handle for back plate
            back_asset = else_assets[-1] if GRID_LINES_NUM == 0 else else_assets[-3]
            back_start_pose = else_start_poses[-1 - GRID_LINES_NUM * 2]
            segmentation_id = 2 # segmentation ID used in segmentation camera sensors
            handle = self.gym.create_actor(
                env_ptr,
                back_asset,
                back_start_pose,
                "white_back",
                self.num_envs * actor_idx + env_idx,
                contact_filter,
                segmentation_id,
            )
            back_color = gymapi.Vec3(1, 1, 1)
            self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, back_color)
            else_handles.append(handle)

            # handle for gridlines
            grid_color = gymapi.Vec3(GRID_COLOR, GRID_COLOR, GRID_COLOR)
            # horizontal
            for i in range(GRID_LINES_NUM):
                grid_h_asset = else_assets[-2]
                grid_h_start_pose = else_start_poses[-GRID_LINES_NUM * 2 + i]
                segmentation_id = 2 # segmentation ID used in segmentation camera sensors
                handle = self.gym.create_actor(
                    env_ptr,
                    grid_h_asset,
                    grid_h_start_pose,
                    "grid_h_%d" % i,
                    self.num_envs * (actor_idx - 3) + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, grid_color)
                else_handles.append(handle)
            # horizontal
            for i in range(GRID_LINES_NUM):
                grid_v_asset = else_assets[-1]
                grid_v_start_pose = else_start_poses[-GRID_LINES_NUM + i]
                segmentation_id = 2 # segmentation ID used in segmentation camera sensors
                handle = self.gym.create_actor(
                    env_ptr,
                    grid_v_asset,
                    grid_v_start_pose,
                    "grid_v_%d" % i,
                    self.num_envs * (actor_idx - 2) + env_idx,
                    contact_filter,
                    segmentation_id,
                )
                self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL, grid_color)
                else_handles.append(handle)

            # light
            l_direction = gymapi.Vec3(0.0, -1.0, 1.0)
            l_color = gymapi.Vec3(1.0, 1.0, 1.0)
            l_ambient = gymapi.Vec3(0.1, 0.1, 0.1)
            self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)

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

    def _build_pd_action_offset_scale(self):
        num_joints = len(DOF_OFFSETS) - 1

        lim_low = self.dof_limits_lower.cpu().numpy()
        lim_high = self.dof_limits_upper.cpu().numpy()

        for j in range(num_joints):
            dof_offset = DOF_OFFSETS[j]
            dof_size = DOF_OFFSETS[j + 1] - DOF_OFFSETS[j]

            if dof_size == 3:
                lim_low[dof_offset : (dof_offset + dof_size)] = -np.pi
                lim_high[dof_offset : (dof_offset + dof_size)] = np.pi

            elif dof_size == 1:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)

                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] = curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)
        self._pd_action_limit_lower = self._pd_action_offset - self._pd_action_scale
        self._pd_action_limit_upper = self._pd_action_offset + self._pd_action_scale

        return

    def _compute_reward(self, actions: torch.Tensor):  # not using
        """allocate reward in self.rew_buf[:]"""
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self._humanoid_contact_forces,
            self._contact_body_ids,
            self._humanoid_rigid_body_pos,
            self.max_episode_length,
            self._enable_early_termination,
            self._termination_height,
        )
        return

    def _refresh_sim_tensors(self, reset=False):
        if not reset:
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_humanoid_obs(env_ids)
        return obs

    def _compute_humanoid_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self._humanoid_root_states
            dof_pos = self._humanoid_dof_pos
            dof_vel = self._humanoid_dof_vel
            key_body_pos = self._humanoid_rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states = self._humanoid_root_states[env_ids]
            dof_pos = self._humanoid_dof_pos[env_ids]
            dof_vel = self._humanoid_dof_vel[env_ids]
            key_body_pos = self._humanoid_rigid_body_pos[env_ids][:, self._key_body_ids, :]

        obs = compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, self._local_root_obs)
        return obs

    def _reset_actors(self, env_ids):
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]

        global_actor_indices = self.global_actor_indices[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._initial_root_states),
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

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        global_indices = self.global_dof_actor_indices.flatten()
        global_indices_tensor = gymtorch.unwrap_tensor(global_indices)

        if not self.cfg["env"]["kinematic"]:
            if self._pd_control:
                pd_tar = self._action_to_pd_targets(self.actions)
                self._target_actions[:, : self.humanoid_num_dof] = pd_tar
                pd_tar_tensor = gymtorch.unwrap_tensor(self._target_actions)
                self.gym.set_dof_position_target_tensor_indexed(
                    self.sim, pd_tar_tensor, global_indices_tensor, len(global_indices)
                )
            else:
                forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
                self._target_actions[:, : self.humanoid_num_dof] = forces
                force_tensor = gymtorch.unwrap_tensor(self._target_actions)
                self.gym.set_dof_actuation_force_tensor_indexed(
                    self.sim, force_tensor, global_indices_tensor, len(global_indices)
                )

        return

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

    def render(self, mode="rgb_array"):
        if self.viewer and self.camera_follow:
            self._update_camera()
        return super().render(mode)

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        # default
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 3.0, 1.0)
        
        # upper view (prior_viz)
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1] - 4.0, 2.5)

        # static
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0], self._cam_prev_char_pos[1], 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self._humanoid_root_states[0, 0:3].cpu().numpy()

        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], char_root_pos[1] + cam_delta[1], cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    # dof_obs_size = 64
    # dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : (dof_offset + dof_size)]

        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_rotation_6d(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset : (dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs


@torch.jit.script
def compute_humanoid_observations(root_states, dof_pos, dof_vel, key_body_pos, local_root_obs):
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


@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


@torch.jit.script
def compute_humanoid_reset(
    reset_buf,
    progress_buf,
    contact_buf,
    contact_body_ids,
    rigid_body_pos,
    max_episode_length,
    enable_early_termination,
    termination_height,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


