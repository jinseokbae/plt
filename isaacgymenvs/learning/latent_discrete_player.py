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

from rl_games.algos_torch import players, torch_ext, model_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs

import isaacgymenvs.learning.mlp_continuous_player as mlp_continuous_player
from gym import spaces
import numpy as np
import pathlib # (dglim - visualization)


DEBUG = False
class LatentDiscretePlayer(mlp_continuous_player.MLPContinuousPlayer):
    def __init__(self, params):
        assert params["load_pretrained"]
        pretrained_path = params["load_pretrained_path"]
        self.load_pretrained_networks(pretrained_path, params["config"]["device"])
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
        if net_config['enc_type'] == "discrete":
            self.pre_model_quantizer = self.pre_model.a2c_network.quantizer
            self.use_continuous_prior = False
        elif net_config['enc_type'] == "hybrid":
            self.pre_model_quantizer = self.pre_model.a2c_network.encoder._post_quantizer
            self.use_continuous_prior = True
        if DEBUG: # testing pretrained encoder
            expert_goal_observation_shape = weights["goal_input_mean_std"]["running_mean"].shape
            expert_goal_input_mean_std = RunningMeanStd(expert_goal_observation_shape)
            expert_goal_input_mean_std.load_state_dict(weights["goal_input_mean_std"])
            self._expert_goal_input_mean_std = expert_goal_input_mean_std.to(device)
            self._expert_goal_input_mean_std.eval()
        # check if group mode or not
        if hasattr(self.pre_model.a2c_network, "num_groups"):
            self._num_quant_groups = self.pre_model.a2c_network.num_groups
            assert self._num_quant_groups > 1
        else:
            self._num_quant_groups = 1
        return

    def get_expert_indices(self, obs): # only used for debug
        with torch.no_grad():
            processed_obs = obs
            goal_obs = self.env._compute_ref_diff_observations()
            processed_goal_obs = self._expert_goal_input_mean_std(goal_obs)
            input_dict = {
                "obs": processed_obs,
                "goal_obs": processed_goal_obs,
            }
            res_dict = self.pre_model.forward_indices(input_dict)
            indices = res_dict["indices"][:self.num_active_quants]
        return indices

    def _setup_action_space(self):
        code_num = self.pre_net_config["code_num"]
        self.code_num = code_num
        quant_type = self.pre_net_config["quant_type"]
        self.pre_model_quant_type = quant_type
        if quant_type in ["rvq", "rvq_cosine", "rqvq"] or self._num_quant_groups > 1:
            num_quants = self.config["num_quants"]
            assert num_quants <= self.pre_net_config["num_quants"]
            num_quants *= self._num_quant_groups
            self.actions_num = [code_num for _ in range(num_quants)]
            self._action_space = spaces.Tuple([spaces.Discrete(n=code_num) for _ in range(num_quants)])
            self.is_multi_discrete = True
            self.num_active_quants = num_quants
        else:
            self.actions_num = code_num
            self._action_space = spaces.Discrete(n=code_num)
            self.is_multi_discrete = False
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
            if is_deterministic:
                if self.is_multi_discrete:
                    stacked_logits = torch.stack(res_dict["logits"], dim=1)
                    indices = stacked_logits.argmax(dim=-1)
                else:
                    indices = res_dict["logits"].argmax(dim=-1)
            else:
                indices = res_dict["actions"]

            if DEBUG: # override
                indices = self.get_expert_indices(_processed_obs)

            # retrieve
            if self._num_quant_groups > 1:
                indices_splits = torch.chunk(indices, self._num_quant_groups, -1)
                latent = []
                for i, split in enumerate(indices_splits):
                    latent_i = self.pre_model_quantizer[i].get_codes_from_indices(split)
                    if self.pre_model_quant_type in ["rvq", "rfsq"]:
                        latent_i = latent_i.sum(dim=0)
                    elif self.pre_model_quant_type in ["simple"]:
                        latent_i = latent_i.squeeze(1)
                    latent.append(latent_i)
                latent = torch.cat(latent, dim=-1)
            else:
                if self.is_multi_discrete:
                    latent = self.pre_model_quantizer.get_codes_from_indices(indices)
                    latent = latent.sum(dim=0)
                else:
                    latent = self.pre_model_quantizer.get_codes_from_indices(indices)
            if self.use_continuous_prior:
                if self.pre_net_config.get("post_cond_prior", False):
                    prior_input_dict = {"obs": _processed_obs, "post_z": latent}
                else:
                    prior_input_dict = {"obs": _processed_obs}
                prior_dict = self.pre_model.forward_prior(prior_input_dict)
                prior_mu = prior_dict['mu']
                latent = latent + prior_mu

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
