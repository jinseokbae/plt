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
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
import os
from omegaconf import ListConfig, OmegaConf
from pathlib import Path
import pathlib # (dglim - visualization)

from rl_games.algos_torch import players, torch_ext, model_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs

import isaacgymenvs.learning.mlp_continuous_player as mlp_continuous_player
from gym import spaces
import numpy as np


class LatentContinuousPlayer(mlp_continuous_player.MLPContinuousPlayer):
    def __init__(self, params):
        assert params["load_pretrained"]
        pretrained_path = params["load_pretrained_path"]
        self.load_pretrained_networks(pretrained_path, params["config"]["device"])
        self._num_active_quants = params["config"].get("num_quants", None)

        # FSQ
        self.use_fsq = params.get("use_fsq", False)

        # Projecter
        self.use_projector = params.get("use_projector", False)

        super().__init__(params)
        return

    def restore(self, fn):
        super().restore(fn)
        # (dglim - visualization)
        self.checkpoint_fn = pathlib.Path(fn).stem
        self.env.checkpoint_fn = self.checkpoint_fn
        return

    def load_pretrained_networks(self, path, device):
        assert os.path.isdir(path)
        params = OmegaConf.load(os.path.join(path, "config.yaml"))
        net_config = dict(OmegaConf.load(os.path.join(path, "net_config.yaml")))
        for cfg_key in net_config.keys():
            if type(net_config[cfg_key]) == ListConfig:
                net_config[cfg_key] = tuple(net_config[cfg_key])

        builder = model_builder.ModelBuilder()
        network = builder.load(params["train"]["params"])
        ckpt_list = list(Path(os.path.join(path, "nn")).rglob("*.pth"))
        try:
            pretrained_ckpt = sorted(ckpt_list, key=lambda x: int(str(x).split("_")[-1][:-4]))[-1]
        except:
            pretrained_ckpt = sorted(ckpt_list)[0]
        print("\033[30m \033[106m" + str(pretrained_ckpt) + "\033[0m")
        weights = torch_ext.load_checkpoint(pretrained_ckpt)
        self.pre_model = network.build(net_config).to(device)
        self.pre_model.load_state_dict(weights["model"])
        self.pre_model.eval()
        self.pre_net_config = net_config
        self.use_continuous_prior = self.pre_net_config["enc_type"] == "continuous" and self.pre_net_config["continuous_enc_style"] == "statecond"
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if self.use_fsq:
            config["fsq_levels"] = self._fsq_levels
        elif self.use_projector:
            config["proj_dim"] = self._proj_dim
            config["num_proj"] = self._num_active_quants
            config["code_num"] = self.pre_net_config["code_num"]
        return config

    def _setup_action_space(self):
        if self.use_fsq:
            codebook_size = self.pre_net_config["code_num"]
            if self.pre_net_config["quant_type"] in ["rvq"]:
                codebook_num = self._num_active_quants
            else:
                codebook_num = 1
            discrete_code_num = [2 ** 10, 2 ** 13]
            assert codebook_size in discrete_code_num
            levels = [
                [16, 8, 8],
                [16, 8, 8, 8]
            ]
            idx = discrete_code_num.index(codebook_size)
            chosen_level = levels[idx]
            self._fsq_levels = chosen_level
            self.actions_num = len(self._fsq_levels) * codebook_num
        elif self.use_projector:
            if self.pre_net_config["quant_type"] in ["rvq"]:
                codebook_num = self._num_active_quants
            else:
                codebook_num = 1
            self._unproj_dim = self.pre_net_config["latent_shape"][-1]
            self._proj_dim = self.config["proj_dim"]
            self.actions_num = self._proj_dim * codebook_num
        else:
            self.actions_num = self.pre_net_config["latent_shape"][-1]
        self._action_space = spaces.Box(np.ones(self.actions_num) * -np.inf, np.ones(self.actions_num) * np.inf)

        # real action dimension in environment
        env_action_space = self.env_info["action_space"]
        self.actions_low = torch.from_numpy(env_action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(env_action_space.high.copy()).float().to(self.device)
        return

    def get_action(self, obs, is_deterministic=False):
        goal_obs = obs["goal_obs"]
        obs = obs["obs"]

        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
            goal_obs = unsqueeze_obs(goal_obs)

        _processed_obs = self._preproc_obs(obs)
        _processed_goal_obs = goal_obs
        processed_obs = torch.cat([_processed_obs, _processed_goal_obs], dim=-1)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
            latent_mu = res_dict["mus"]
            latent_action = res_dict["actions"]

            # whether using prior or not
            if self.use_continuous_prior:
                prior_input_dict = {"obs": _processed_obs}
                prior_dict = self.pre_model.forward_prior(prior_input_dict)
                prior_mu = prior_dict['mu']
                latent_mu = latent_mu + prior_mu
                latent_action = latent_action + prior_mu

            if is_deterministic:
                latent = latent_mu
            else:
                latent = latent_action
            pre_model_dict = dict(obs=_processed_obs, latent=latent)
            res_dict["decoded_actions"], res_dict["decoded_mus"] = self.pre_model.decode(**pre_model_dict)
        mu = res_dict["decoded_mus"]
        action = res_dict["decoded_actions"]
        self.states = res_dict["rnn_states"]
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return players.rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action
