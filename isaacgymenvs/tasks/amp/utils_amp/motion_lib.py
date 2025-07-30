# Copyright (c) 2018-2022, NVIDIA Corporation
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
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os

import numpy as np
import torch
import yaml
from tqdm import tqdm

from isaacgymenvs.utils.torch_jit_utils import *

from ..poselib.poselib.core.rotation3d import *
from ..poselib.poselib.skeleton.skeleton3d import SkeletonMotion

# from isaacgym.torch_utils import *

USE_CACHE = True
print("MOVING MOTION DATA TO GPU, USING CACHE:", USE_CACHE)

if not USE_CACHE:
    old_numpy = torch.Tensor.numpy

    class Patch:
        def numpy(self):
            if self.is_cuda:
                return self.to("cpu").numpy()
            else:
                return old_numpy(self)

    torch.Tensor.numpy = Patch.numpy


class DeviceCache:
    def __init__(self, obj, device):
        self.obj = obj
        self.device = device

        keys = dir(obj)
        num_added = 0
        for k in keys:
            try:
                out = getattr(obj, k)
            except:
                # print("Error for key=", k)
                continue

            if isinstance(out, torch.Tensor):
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1
            elif isinstance(out, np.ndarray):
                out = torch.tensor(out)
                if out.is_floating_point():
                    out = out.to(self.device, dtype=torch.float32)
                else:
                    out.to(self.device)
                setattr(self, k, out)
                num_added += 1

        # print("Total added", num_added)

    def __getattr__(self, string):
        out = getattr(self.obj, string)
        return out


class MotionLib:
    def __init__(self, motion_file, dof_body_ids, dof_offsets, key_body_ids, device, min_len, motion_matching, generator=None):
        self._dof_body_ids = dof_body_ids
        self._dof_offsets = dof_offsets
        self._num_dof = dof_offsets[-1]
        self._key_body_ids = torch.tensor(key_body_ids, device=device)
        self._device = device
        self._min_len = min_len
        self._eval_motion_matching = motion_matching
        self._generator = generator
        self._load_motions(motion_file)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        self.motion_ids = torch.arange(len(self._motion_lengths), dtype=torch.long, device=self._device)

        return

    def num_motions(self):
        return len(self._motion_lengths)

    def get_total_length(self):
        return sum(self._motion_lengths)

    def sample_motions(self, n):
        motion_ids = torch.multinomial(self._motion_weights, num_samples=n, replacement=True, generator=self._generator)

        # m = self.num_motions()
        # motion_ids = np.random.choice(m, size=n, replace=True, p=self._motion_weights)
        # motion_ids = torch.tensor(motion_ids, device=self._device, dtype=torch.long)
        return motion_ids

    def sample_time(self, motion_ids, truncate_time=None):
        n = len(motion_ids)
        phase = torch.rand(motion_ids.shape, device=self._device, generator=self._generator)

        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time

    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]

    def get_motion_state(self, motion_ids, motion_times, output_global=False):
        n = len(motion_ids)
        num_bodies = self._get_num_bodies()
        num_key_bodies = self._key_body_ids.shape[0]

        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel = self.grvs[f0l]

        root_ang_vel = self.gravs[f0l]

        key_pos0 = self.gts[f0l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self._key_body_ids.unsqueeze(0)]

        dof_vel = self.dvs[f0l]

        vals = [root_pos0, root_pos1, local_rot0, local_rot1, root_vel, root_ang_vel, key_pos0, key_pos1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1

        root_rot = slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1

        local_rot = slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos = self._local_rotation_to_dof(local_rot)

        if output_global:
            # global positions
            global_pos0 = self.gts[f0l]
            global_pos1 = self.gts[f1l]
            global_pos = (1.0 - blend_exp) * global_pos0 + blend_exp * global_pos1

            # global rigid body joint rotations
            global_rot0 = self.grs[f0l]
            global_rot1 = self.grs[f1l]
            global_rot = slerp(global_rot0, global_rot1, torch.unsqueeze(blend, axis=-1))

            # global rigid body velocity
            global_vel = self.gvs[f0l]

            # global rigid body joint angular velocities
            global_ang_vel = self.gavs[f0l]
        else:
            global_pos, global_rot, global_vel, global_ang_vel = None, None, None, None
        return (
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
        )

    def _load_motions(self, motion_file):
        self._motion_lengths = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []

        gts = []
        grs = []
        lrs = []
        grvs = []
        gravs = []
        gvs = []
        gavs = []
        dvs = []

        total_len = 0.0

        motion_files, motion_weights = self._fetch_motion_files(motion_file)
        num_motion_files = len(motion_files)
        with tqdm(range(num_motion_files)) as pbar:
            motion_root_dir = "/".join(
                str(motion_files[0]).split("/")[: str(motion_files[0]).split("/").index("motions") + 1]
            )
            cache_root_dir = os.path.join(motion_root_dir, ".cache")
            for f in pbar:
                pbar.set_description("Loaded motions - %d/%d" % (f + 1, num_motion_files))
                curr_file = str(motion_files[f])
                if num_motion_files < 10:
                    print("Loading {:d}/{:d} motion files: {:s}".format(f + 1, num_motion_files, str(curr_file)))
                # find cache
                _file_name = "/".join(curr_file.split("/")[curr_file.split("/").index("motions") + 1 :])
                _file_name = ".".join(_file_name.split(".")[:-1] + ["pt"])
                _cached_file = os.path.join(cache_root_dir, _file_name)
                if os.path.isfile(_cached_file):
                    curr_motion = torch.load(_cached_file)
                else:
                    curr_motion = SkeletonMotion.from_file(curr_file)
                    curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
                    curr_motion.dof_vels = curr_dof_vels

                    parent_dir = "/".join(_cached_file.split("/")[:-1])
                    os.makedirs(parent_dir, exist_ok=True)
                    torch.save(curr_motion, _cached_file)
                    print("Cached :", _file_name)

                if not hasattr(self, "num_bodies"):
                    self.num_bodies = curr_motion.num_joints
                motion_fps = int(curr_motion.fps)
                curr_dt = 1.0 / motion_fps

                num_frames = curr_motion.tensor.shape[0]
                curr_len = 1.0 / motion_fps * (num_frames - 1)
                if curr_len < self._min_len:
                    continue

                self._motion_fps.append(motion_fps)
                self._motion_dt.append(curr_dt)
                self._motion_num_frames.append(num_frames)

                # Moving motion tensors to the GPU
                if USE_CACHE:
                    curr_motion = DeviceCache(curr_motion, self._device)
                else:
                    curr_motion.tensor = curr_motion.tensor.to(self._device)
                    curr_motion._skeleton_tree._parent_indices = curr_motion._skeleton_tree._parent_indices.to(
                        self._device
                    )
                    curr_motion._skeleton_tree._local_translation = curr_motion._skeleton_tree._local_translation.to(
                        self._device
                    )
                    curr_motion._rotation = curr_motion._rotation.to(self._device)

                gts.append(curr_motion.global_translation)
                grs.append(curr_motion.global_rotation)
                lrs.append(curr_motion.local_rotation)
                grvs.append(curr_motion.global_root_velocity)
                gravs.append(curr_motion.global_root_angular_velocity)
                gvs.append(curr_motion.global_velocity)
                gavs.append(curr_motion.global_angular_velocity)
                dvs.append(curr_motion.dof_vels)
                del curr_motion

                self._motion_lengths.append(curr_len)

                curr_weight = motion_weights[f]
                self._motion_weights.append(curr_weight)
        
        # stats
        self.gts = torch.cat(gts, dim=0)
        self.grs = torch.cat(grs, dim=0)
        self.lrs = torch.cat(lrs, dim=0)
        self.grvs = torch.cat(grvs, dim=0)
        self.gravs = torch.cat(gravs, dim=0)
        self.gvs = torch.cat(gvs, dim=0)
        self.gavs = torch.cat(gavs, dim=0)
        self.dvs = torch.cat(dvs, dim=0)
        
        self._motion_lengths = torch.tensor(self._motion_lengths, device=self._device, dtype=torch.float32)

        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float32, device=self._device)
        self._motion_weights /= self._motion_weights.sum()

        self._motion_fps = torch.tensor(self._motion_fps, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(self._motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, device=self._device)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

        # motion matching
        if self._eval_motion_matching: # assume all the fps are the same
            # to validate
            assert (self._motion_fps - self._motion_fps.mean()).int().abs().sum() == 0

            # calculate dof pos
            dps = self._local_rotation_to_dof(self.lrs)
            root_pos = self.gts[:, 0]
            root_rot = self.grs[:, 0]
            root_vel = self.grvs
            root_ang_vel = self.gravs
            root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
            dof_pos = dps
            dof_vel = self.dvs
            key_body_pos = self.gts[:, self._key_body_ids]
            feats = build_amp_observations(
                root_states,
                dof_pos,
                dof_vel,
                key_body_pos,
                True
            )
            self.feats_mean, self.feats_std = feats.mean(dim=0, keepdim=True), feats.std(dim=0, keepdim=True)
            norm_feats = (feats - self.feats_mean) / self.feats_std

            hist_feats = []
            curr_feats = []
            transition_lengths = []

            lengths = self._motion_num_frames
            lengths_shifted = lengths.roll(1)
            lengths_shifted[0] = 0
            length_starts = lengths_shifted.cumsum(0)
            for k in range(self.num_motions()):
                offset = length_starts[k]
                span = self._motion_num_frames[k]

                curr_feats_k = norm_feats[offset + 1:offset+span]
                hist_feats_k = norm_feats[offset:offset+span - 1]

                curr_feats.append(curr_feats_k)
                hist_feats.append(hist_feats_k)
                transition_lengths.append(curr_feats_k.shape[0])

            self.curr_feats = torch.cat(curr_feats, dim=0)
            self.hist_feats = torch.cat(hist_feats, dim=0)
            self.transition_lengths = torch.tensor(transition_lengths, dtype=torch.int64, device=self._device)
            lengths_shifted = self.transition_lengths.roll(1)
            lengths_shifted[0] = 0
            self.transition_length_starts = lengths_shifted.cumsum(0)
        return

    def _fetch_motion_files(self, motion_file):
        if isinstance(motion_file, list):
            motion_files = motion_file  # [motion_file]
            motion_weights = [1.0 for _ in range(len(motion_files))]
        else:
            ext = os.path.splitext(motion_file)[1]
            if ext == ".yaml":
                dir_name = os.path.dirname(motion_file)
                motion_files = []
                motion_weights = []

                with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                    motion_config = yaml.load(f, Loader=yaml.SafeLoader)

                motion_list = motion_config["motions"]
                for motion_entry in motion_list:
                    curr_file = motion_entry["file"]
                    curr_weight = motion_entry["weight"]
                    assert curr_weight >= 0

                    curr_file = os.path.join(dir_name, curr_file)
                    motion_weights.append(curr_weight)
                    motion_files.append(curr_file)
            else:
                raise NotImplementedError()

        return motion_files, motion_weights

    def _calc_frame_blend(self, time, len, num_frames, dt):
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        return self.num_bodies

    def _compute_motion_dof_vels(self, motion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)

        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels

    def _local_rotation_to_dof(self, local_rot):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self._num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_q = local_rot[:, body_id]
                joint_exp_map = quat_to_exp_map(joint_q)
                dof_pos[:, joint_offset : (joint_offset + joint_size)] = joint_exp_map
            elif joint_size == 1:
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = quat_to_angle_axis(joint_q)
                joint_theta = joint_theta * joint_axis[..., 1]  # assume joint is always along y axis

                joint_theta = normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert False

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self._dof_body_ids
        dof_offsets = self._dof_offsets

        dof_vel = torch.zeros([self._num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset : (joint_offset + joint_size)] = joint_vel

            elif joint_size == 1:
                assert joint_size == 1
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[1]  # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert False

        return dof_vel

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

