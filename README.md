# PLT (SIGGRAPH 2025)
This is an official implementation for SIGGRAPH 2025 paper titled "PLT: Part-Wise Latent Tokens as Adaptable Motion Priors for Physically Simulated Characters".
We provide codes for all the training and testing environments demonstrated in the main paper.
This code repo is largely based on our pervious project [`hybrid_latent_prior`](https://github.com/jinseokbae/hybrid_latent_prior).
- Paper : [link](https://dl.acm.org/doi/10.1145/3721238.3730637)
- Video : [link](https://www.youtube.com/watch?v=dSHMMwQ9GHE)
- Project Page : [link](https://jinseokbae.github.io/plt)


## 🛠️ Installation
This code is based on [Isaac Gym Preview 4](https://developer.nvidia.com/isaac-gym).
Please run installation code and create a conda environment following the instruction in Isaac Gym Preview 4.
We assume the name of conda environment is `plt` (hybrid latent representation).
Then, run the following script.

```shell
conda activate plt
cd plt
pip install -e .
bash setup_conda_env_extra_cuda117.sh
# When older cuda version is only available
# bash setup_conda_env_extra_cuda111.sh
```

## 💾 Data Preparation
In the paper, we mainly use three datasets: [LaFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), [AMASS](https://amass.is.tue.mpg.de/), and [Assassin Moves](https://www.reallusion.com/ContentStore/iclone/Pack/assassin_moves/default.html) dataset.
We retargeted each motion capture data to humanoid model in this repo.
For AMASS dataset, we directly utilize [PULSE](https://github.com/ZhengyiLuo/PULSE) codebase, so please refer to the instruction to the original repo.

Due to the license issue, we cannot distribute the whole dataset retargeted to the humanoid. 
We strongly recommend you to retarget original LaFAN1 motions using codes under following our [instruction](isaacgymenvs/tasks/amp/poselib/README.md).
> ⚠️ **Dataset Usage Notice.**<br>
> This project uses the LaFAN1 dataset, which is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 License (CC BY-NC-ND 4.0).
> As such, we do not redistribute any modified or retargeted versions of the dataset.
> Users must download the original data from the official source and run our provided script to generate compatible input.

If you have any problem, please reach us via e-mail.

## 📚 Pretrained Models
We share the pretrained weights for the policies.
Please download from this [Google Drive link](https://drive.google.com/drive/folders/10kn_lUJIYkQijNj2nWryB9gRsMUmfF3e?usp=drive_link) and unzip the folder under `isaacgymenvs/`.
We assume all the pretrained polices are located under `isaacgymenvs/pretrained/`.
To run and visualize the result, please run the following commands under `isaacgmynevs/`.
```shell
cd isaacgymenvs/
```
Please note that these commands are using sample reference motions listed in `assets/motions/samples/` for test (unseen motions during the training).

### 🏃‍➡️ Part 1: Imitation Policies
To train our PLT imitation policy, we employ online distillation suggested by [PULSE](https://arxiv.org/abs/2310.04582).
We empirically found that this strategy allows stable training of complex student policies that are usually employing additional loss terms for regularizing latent space.

**(a) Expert Imitation Policy** 

We provide a simple expert policy trained through [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (PPO) and [Adversarial Motion Prior](https://dl.acm.org/doi/10.1145/3450626.3459670).
```shell
python train.py test=True num_envs=1 task=Imitation train=LafanImitation/LafanExpertAMPPPO motion_dataset=samples checkpoint=pretrained/expert_lafan_imitation/expert_lafan_imitation_50000.pth
```

**(b) PLT Imitation Policy**

We share a PLT-5 model, where agent has 5 body parts (trunk, left/right arms, left/right legs).
```shell
python train.py test=True num_envs=1 task=Imitation train=LafanImitation/LafanPLTDistill motion_dataset=samples checkpoint=pretrained/plt5_lafan_imitation/plt5_lafan_imitation_25000.pth
```

### 📝 Part 2: Task Policies

## Run
TODO

## License
This repository contains three types of code:
1. Code originally authored by NVIDIA (Isaac Gym), licensed under the [BSD 3-Clause License](third_party/LICENSE.txt).
2. PyTorch implementaion on the various vector quantization methods `vector_quantize_pytorch`(https://github.com/lucidrains/vector-quantize-pytorch), licensed under the [MIT License](https://github.com/lucidrains/vector-quantize-pytorch?tab=MIT-1-ov-file).
3. Code authored by ourselves, licensed under the [MIT License](LICENSE).