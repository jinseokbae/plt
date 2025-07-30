# PLT (SIGGRAPH 2025)
This is an official implementation for SIGGRAPH 2025 paper titled "PLT: Part-Wise Latent Tokens as Adaptable Motion Priors for Physically Simulated Characters".
We provide codes for all the training and testing environments demonstrated in the main paper.
This code repo is largely based on our pervious project [`hybrid_latent_prior`](https://github.com/jinseokbae/hybrid_latent_prior).
- Paper : [link](https://dl.acm.org/doi/10.1145/3721238.3730637)
- Video : [link](https://www.youtube.com/watch?v=dSHMMwQ9GHE)
- Project Page : [link](https://jinseokbae.github.io/plt)


## Installation
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

## Data Preparation
In the paper, we mainly use three datasets: [LaFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset), [AMASS](https://amass.is.tue.mpg.de/), and [Assassin Moves](https://www.reallusion.com/ContentStore/iclone/Pack/assassin_moves/default.html) dataset.
We retargeted each motion capture data to humanoid model in this repo.
For AMASS dataset, we directly utilize [PULSE](https://github.com/ZhengyiLuo/PULSE) codebase, so please refer to the instruction to the original repo.

Due to the license issue, we cannot distribute the whole dataset retargeted to the humanoid. 
We strongly recommend you to retarget original LaFAN1 motions using codes under following our [instruction](isaacgymenvs/tasks/amp/poselib/README.md).
If you have any problem or want to get pre-processed data rapidly, please reach us via e-mail.

## Pretrained Models
TODO

## Run
TODO

## License
This repository contains three types of code:
1. Code originally authored by NVIDIA (Isaac Gym), licensed under the [BSD 3-Clause License](third_party/LICENSE.txt).
2. PyTorch implementaion on the various vector quantization methods `vector_quantize_pytorch`(https://github.com/lucidrains/vector-quantize-pytorch), licensed under the [MIT License](https://github.com/lucidrains/vector-quantize-pytorch?tab=MIT-1-ov-file).
3. Code authored by ourselves, licensed under the [MIT License](LICENSE).