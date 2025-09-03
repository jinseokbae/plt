# Retargeting LaFAN1 Dataset on AMP Humanoid

> ⚠️ **Important Note.**<br>
> [Updated Sep. 2, 2025] There are some missing configuration files in the current version. As a temporal solution to this, please directly contact me (`capoo95 at snu dot ac dot kr`) if you need a retargeted LaFAN1 that is ready for training.

## 1. Download BVH files
- Please download bvh files from the original LaFAN1 [code repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
> ⚠️ **For Better Quality.**<br>
> We also recommend to use [revised version of LaFAN1](https://github.com/orangeduck/lafan1-resolved).
- Then place bvh files under `lafan1/data`.

## 2. Convert BVH files to NPZ format
- Run `lafan_bvh_to_npz.py`.
- processed files will be stored under the folder namaed `lafan1/lafan1_npz_YEAR-MONTH-DATE` (*e.g. lafan1_npz_2024-Nov-20*).
- create a symbolic link for retargeting
```shell
mkdir data/LAFAN && cd data/LAFAN
ln -s ../../../lafan1/lafan1_npz_YEAR-MONTH-DATE .
cd ../../../
```

## 3. Retarget motions
- `mkdir data/AMP`
- Run `lafan_processing.py`.
- Then the retargeted motions (npy files) will be stored under the folder named as `data/AMP/LAFAN_ALL_YEAR-MONTH-DATE`.
- make symbolic link as
```shell
cd PATH_TO_THE_ROOT/assets/amp/motions
ln -s PATH_TO_THE_ROOT/isaacgymenvs/tasks/amp/poselib/data/AMP/LAFAN_ALL_YEAR-MONTH-DATE LAFAN_ALL
```
- Now all set!