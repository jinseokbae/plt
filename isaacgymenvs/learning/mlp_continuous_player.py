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
from rl_games.algos_torch import players, torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer
from rl_games.common.tr_helpers import unsqueeze_obs

import isaacgymenvs.learning.common_player as common_player


class MLPContinuousPlayer(common_player.CommonPlayer):
    def __init__(self, params):
        super().__init__(params)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if hasattr(self, "env"):
            goal_input_shape = self.env.goal_observation_space.shape
        else:
            goal_input_shape = self.env_info["goal_observation_space"]
        config["input_shape"] = (config["input_shape"][-1] + goal_input_shape[-1],)
        return config

    def obs_to_torch(self, obs):
        upd_obs_dict = super().obs_to_torch(obs)
        if isinstance(obs, dict):
            if "goal_obs" in obs:
                goal_obs = obs["goal_obs"]
            if isinstance(goal_obs, dict):
                upd_goal_obs = {}
                for key, value in goal_obs.items():
                    upd_goal_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_goal_obs = self.cast_obs(goal_obs)
        else:
            upd_goal_obs = None
        upd_obs_dict["goal_obs"] = upd_goal_obs
        return upd_obs_dict

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
        mu = res_dict["mus"]
        action = res_dict["actions"]
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
