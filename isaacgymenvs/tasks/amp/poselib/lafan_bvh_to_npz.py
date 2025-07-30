import os
from pathlib import Path
import numpy as np
from lafan1.extract import *
import torch
# using poselib from isaacgymenvs
from poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from poselib.core.rotation3d import *
from poselib.visualization.common import (
    plot_skeleton_state,
    plot_skeleton_motion_interactive,
)
from tqdm import tqdm
from datetime import datetime, timedelta, timezone

train = True
debug = False

if __name__ == "__main__":
    bvh_dir = "lafan1/data"
    bvh_files = list(Path(bvh_dir).rglob("*.bvh"))
    if train:
        actors = ['subject1', 'subject2', 'subject3', 'subject4']
    else:
        actors = ['subject5']

    date_str = datetime.now(timezone(timedelta(hours=9))).strftime("_%Y-%b-%d")
    save_dir = "../lafan1_npz" + date_str
    split = "train" if train else "test"
    save_dir = os.path.join(save_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    for bvh_file in tqdm(bvh_files):
        seq_name, subject = ntpath.basename(str(bvh_file)[:-4]).split('_')
        if subject in actors:
            if debug:
                print("Processing :", bvh_file)
            if bvh_dir == "lafan1/data":
                anim, frametime = read_modified_bvh(str(bvh_file))
            else:
                anim, frametime = read_bvh(str(bvh_file))
            fps = round(1 / frametime)
            if debug:
                print("FPS :", fps)
            assert fps % 10 == 0
            # frames
            global_translation = torch.from_numpy(anim.pos)
            local_rotation = torch.from_numpy(anim.quats)
            local_rotation = torch.cat([local_rotation[..., 1:], local_rotation[..., :1]], dim=-1)

            # basic info
            node_names = anim.bones
            parent_indices = torch.from_numpy(anim.parents)
            local_translation = torch.from_numpy(anim.offsets)

            # toe modification
            initial_lfoot_quat = local_rotation[0, node_names.index("LeftFoot")].clone()
            initial_rfoot_quat = local_rotation[0, node_names.index("RightFoot")].clone()
            local_rotation[:, node_names.index("LeftFoot")] = quat_mul(
                quat_inverse(initial_lfoot_quat), 
                local_rotation[:, node_names.index("LeftFoot")]
            )
            local_rotation[:, node_names.index("RightFoot")] = quat_mul(
                quat_inverse(initial_rfoot_quat),
                local_rotation[:, node_names.index("RightFoot")]
            )
            local_translation[node_names.index("LeftToe")] = quat_rotate(initial_lfoot_quat, local_translation[node_names.index("LeftToe")])
            local_translation[node_names.index("RightToe")] = quat_rotate(initial_rfoot_quat, local_translation[node_names.index("RightToe")])
            # create skeleton_tree
            skeleton_tree = SkeletonTree(node_names, parent_indices, local_translation)

            # make skeleton motion
            skeleton_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree=skeleton_tree,
                r=local_rotation,
                t=global_translation[:, node_names.index('Hips')],
                is_local=True
            )
            skeleton_motion = SkeletonMotion.from_skeleton_state(
                skeleton_state=skeleton_state,
                fps=fps
            )

            # compute local angular velocity
            local_rot0 = skeleton_motion.local_rotation[:-1]
            local_rot1 = skeleton_motion.local_rotation[1:]

            diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
            diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
            local_angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / frametime
            local_angular_velocity = torch.cat([local_angular_velocity, local_angular_velocity[-1:]], dim=0)
            skeleton_motion.local_angular_velocity = local_angular_velocity

            ret = dict(
                root_translation=skeleton_motion.global_translation[:, node_names.index('Hips')], # (T, 3) 

                root_rotation=skeleton_motion.global_rotation[:, node_names.index('Hips')], # (T, 3) # 

                root_velocity=skeleton_motion.global_root_velocity, # (T, 3) # heading canonicalization
                # my_quat_rotate(calc_heading_quat_inv(root_rotation), root_velocity)

                root_angular_velocity=skeleton_motion.global_root_angular_velocity, # (T, 3)

                # rotation
                local_rotation=skeleton_motion.local_rotation, # (T, 22, 4)

                local_angular_velocity=skeleton_motion.local_angular_velocity, # (T, 22, 3)

                global_translation=skeleton_motion.global_translation, # (T, 22, 3) # heading canonicalization
                # part_indices = [node_names.index(part_name) for part_name in part_names]
                # my_quat_rotate(calc_heading_quat_inv(root_rotation), global_transation[:, part_indices] - root_translation[:, None])

                global_velocity=skeleton_motion.global_velocity, # (T, 22, 3) # heading canonicalization
                # part_indices = [node_names.index(part_name) for part_name in part_names]
                # my_quat_rotate(calc_heading_quat_inv(root_rotation), global_velocity[:, part_indices] - root_velocity[:, None])
            )
            if debug:
                for key, value in ret.items():
                    print(key, value.shape)
            
            ret.update(
                dict(
                    node_names=node_names,
                    parent_indices=anim.parents,
                    joint_offsets=anim.offsets,
                    fps=fps,
                )
            )

            for key, value in ret.items():
                if torch.is_tensor(value):
                    if value.device != "cpu":
                        value = value.detach()
                    ret[key] = value.cpu().numpy()
            
            # save
            save_file = ntpath.basename(str(bvh_file)[:-4]) + '.npz'
            np.savez(os.path.join(save_dir, save_file), ret)

            if debug:
                # zero pose
                zero_pose = SkeletonState.zero_pose(skeleton_tree)
                plot_skeleton_state(zero_pose)
                # motion
                plot_skeleton_motion_interactive(skeleton_motion)