import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from poselib.core.rotation3d import *
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
from poselib.visualization.common import plot_skeleton_motion_interactive, plot_skeleton_state
from retarget_motion import project_joints
from tqdm import tqdm

VISUALIZE = False
RETARGET_CONFIG = "data/configs/retarget_lafan_to_amp.json"
LAFAN_NPZ_ROOT_DIR = "data/LAFAN/lafan1_npz_2024-Nov-20"  # need to be processed from bvh
OUT_TYPE = "npy"

SUBSET = {
    "loco": ["run", "walk", "sprint"],
    "jumps": ["jumps"]}

MODE = "all"


def main():
    # source npz file path
    splits = ["train", "test"]

    # init - retarget
    with open(RETARGET_CONFIG) as f:
        print("Load config file : %s" % RETARGET_CONFIG)
        retarget_data = json.load(f)
    # load and visualize t-pose files
    target_tpose = SkeletonState.from_file(retarget_data["target_tpose"])
    if VISUALIZE:
        plot_skeleton_state(target_tpose)
    rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

    # loop
    date_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%b-%d")
    for split in splits:
        train = split == "train"
        if OUT_TYPE == "npz":
            save_root_dir = "data/AMP_npz/LAFAN_%s_%s/%s" % (MODE.upper(), date_str, split)
        else:
            save_root_dir = "data/AMP/LAFAN_%s_%s/%s" % (MODE.upper(), date_str, split)

        if train:
            actors = ["subject1", "subject2", "subject3", "subject4"]
        else:
            actors = ["subject5"]
        npz_files = list(Path(os.path.join(LAFAN_NPZ_ROOT_DIR, split)).rglob("*.npz"))

        os.makedirs(save_root_dir, exist_ok=True)
        for npz_file in tqdm(npz_files):
            seq_name, subject = os.path.basename(str(npz_file)[:-4]).split("_")
            if MODE != "all":
                seq_selected = False
                for seq_base_name in SUBSET[MODE]:
                    if seq_base_name in seq_name:
                        seq_selected = True
                        break
            else:
                seq_selected = True

            if (subject in actors) and seq_selected:
                print("Processing %s set" % split, npz_file)
                source_motion = SkeletonMotion.from_lafan_npz(lafan_npz_fname=npz_file)

                skeleton = source_motion.skeleton_tree
                zero_pose = SkeletonState.zero_pose(skeleton)
                local_rotation = zero_pose.local_rotation
                local_rotation[skeleton.index("Hips")] = quat_mul(
                    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([1.0, 0.0, 0.0]), degree=True),
                    local_rotation[skeleton.index("Hips")],
                )
                local_rotation[skeleton.index("Hips")] = quat_mul(
                    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
                    local_rotation[skeleton.index("Hips")],
                )

                local_rotation[skeleton.index("LeftArm")] = quat_mul(
                    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, 1.0]), degree=True),
                    local_rotation[skeleton.index("LeftArm")],
                )
                local_rotation[skeleton.index("RightArm")] = quat_mul(
                    quat_from_angle_axis(angle=torch.tensor([90.0]), axis=torch.tensor([0.0, 0.0, -1.0]), degree=True),
                    local_rotation[skeleton.index("RightArm")],
                )
                source_tpose = zero_pose
                if VISUALIZE:
                    plot_skeleton_state(source_tpose)

                # if VISUALIZE:
                #     plot_skeleton_motion_interactive(source_motion)

                # parse data from retarget config
                joint_mapping = retarget_data["joint_mapping"]
                rotation_to_target_skeleton = torch.tensor(retarget_data["rotation"])

                # run retargeting
                target_motion = source_motion.retarget_to_by_tpose(
                    joint_mapping=retarget_data["joint_mapping"],
                    source_tpose=source_tpose,
                    target_tpose=target_tpose,
                    rotation_to_target_skeleton=rotation_to_target_skeleton,
                    scale_to_target_skeleton=retarget_data["scale"],
                )

                # keep frames between [trim_frame_beg, trim_frame_end - 1]
                frame_beg = retarget_data["trim_frame_beg"]
                frame_end = retarget_data["trim_frame_end"]
                if frame_beg == -1:
                    frame_beg = 0

                if frame_end == -1:
                    frame_end = target_motion.local_rotation.shape[0]

                local_rotation = target_motion.local_rotation
                root_translation = target_motion.root_translation
                local_rotation = local_rotation[frame_beg:frame_end, ...]
                root_translation = root_translation[frame_beg:frame_end, ...]

                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    target_motion.skeleton_tree, local_rotation, root_translation, is_local=True
                )
                target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)

                # need to convert some joints from 3D to 1D (e.g. elbows and knees)
                target_motion = project_joints(target_motion)

                # move the root so that the feet are on the ground
                local_rotation = target_motion.local_rotation
                root_translation = target_motion.root_translation
                tar_global_pos = target_motion.global_translation
                min_h = torch.min(tar_global_pos[..., 2])
                root_translation[:, 2] += -min_h

                # adjust the height of the root to avoid ground penetration
                root_height_offset = retarget_data["root_height_offset"]
                root_translation[:, 2] += root_height_offset

                new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    target_motion.skeleton_tree, local_rotation, root_translation, is_local=True
                )
                target_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_motion.fps)
                body_height = target_motion.global_translation[:, :, 2]
                min_body_height = body_height.min()
                if min_body_height > 0.05:
                    target_motion.root_translation[..., 2] -= min_body_height - 0.05

                # visualize retargeted motion
                if VISUALIZE:
                    plot_skeleton_motion_interactive(target_motion)

                # # save retargeted motion
                if OUT_TYPE == "npy":
                    save_path = os.path.join(save_root_dir, seq_name + "_" + subject + ".npy")
                    target_motion.to_file(save_path)
                elif OUT_TYPE == "npz":
                    keys = [
                        "root_translation",
                        "root_rotation",
                        "root_velocity",
                        "root_angular_velocity",
                        "local_rotation",
                        "local_angular_velocity",
                        "global_translation",
                        "global_velocity",
                        "node_names",
                        "parent_indices",
                        "joint_offsets",
                        "fps",
                    ]
                    npz_dict = dict()
                    npz_dict["global_translation"] = target_motion.global_translation.cpu().numpy()
                    npz_dict["global_rotation"] = target_motion.global_rotation.cpu().numpy()
                    npz_dict["global_velocity"] = target_motion.global_velocity.cpu().numpy()
                    npz_dict["global_angular_velocity"] = target_motion.global_angular_velocity.cpu().numpy()

                    npz_dict["root_translation"] = target_motion.root_translation.cpu().numpy()
                    npz_dict["root_rotation"] = target_motion.global_root_rotation.cpu().numpy()
                    npz_dict["root_velocity"] = target_motion.global_root_velocity.cpu().numpy()
                    npz_dict["root_angular_velocity"] = target_motion.global_root_angular_velocity.cpu().numpy()

                    npz_dict["local_rotation"] = target_motion.local_rotation.cpu().numpy()
                    local_rot = target_motion.local_rotation.clone()
                    local_rot0 = local_rot[:-1]
                    local_rot1 = local_rot[1:]
                    diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
                    diff_quat_data = torch.cat([diff_quat_data, diff_quat_data[-1:]], dim=0)
                    diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
                    local_ang_vel = diff_axis * diff_angle.unsqueeze(-1) / (1 / target_motion.fps)
                    npz_dict["local_angular_velocity"] = local_ang_vel.cpu().numpy()

                    npz_dict["node_names"] = target_motion.skeleton_tree.node_names
                    npz_dict["parent_indices"] = target_motion.skeleton_tree.parent_indices.cpu().numpy()
                    npz_dict["joint_offsets"] = target_motion.skeleton_tree.local_translation.cpu().numpy()
                    npz_dict["fps"] = target_motion.fps

                    save_path = os.path.join(save_root_dir, seq_name + "_" + subject + ".npz")
                    np.savez(save_path, npz_dict)


if __name__ == "__main__":
    main()
