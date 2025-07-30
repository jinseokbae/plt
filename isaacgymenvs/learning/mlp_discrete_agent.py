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

import copy
import time
from datetime import datetime

import numpy as np
import torch
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common, schedulers, vecenv
from tensorboardX import SummaryWriter
from torch import nn, optim

import isaacgymenvs.learning.common_discrete_agent as common_discrete_agent
import isaacgymenvs.learning.replay_buffer as replay_buffer
from isaacgymenvs.utils.torch_jit_utils import to_torch


class MLPDiscreteAgent(common_discrete_agent.CommonDiscreteAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        if self._prioritized_sampling:
            self._normalized_motion_returns = torch.zeros_like(self.vec_env.env._motion_lib._motion_lengths)
        return

    def init_tensors(self):
        super().init_tensors()
        self._build_goal_buffers()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._prioritized_sampling:
            state["normalized_motion_returns"] = self._normalized_motion_returns
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._prioritized_sampling:
            self._normalized_motion_returns = weights["normalized_motion_returns"]
        return

    def get_action_values(self, obs):
        _processed_obs = self._preproc_obs(obs["obs"])
        _processed_goal_obs = obs["goal_obs"]
        processed_obs = torch.cat([_processed_obs, _processed_goal_obs], dim=-1)
        self.model.eval()
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.rnn_states,
        }

        with torch.no_grad():
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_central_value(input_dict)
                res_dict["values"] = value
        return res_dict

    def _eval_critic(self, obs_dict):
        self.model.eval()
        obs = obs_dict["obs"]
        goal_obs = obs_dict["goal_obs"]

        _processed_obs = self._preproc_obs(obs)
        _processed_goal_obs = goal_obs
        processed_obs = torch.cat([_processed_obs, _processed_goal_obs], dim=-1)

        if self.normalize_input:
            processed_obs = self.model.norm_obs(processed_obs)

        value = self.model.a2c_network.eval_critic(processed_obs)

        if self.normalize_value:
            value = self.value_mean_std(value, True)
        return value

    def play_steps(self):
        self.set_eval()

        epinfos = []
        update_list = self.update_list

        for n in range(self.horizon_length):
            self.obs, done_env_ids = self._env_reset_done()
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            self.experience_buffer.update_data("goal_obses", n, self.obs["goal_obs"])

            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])

            if self.has_central_value:
                self.experience_buffer.update_data("states", n, self.obs["states"])

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)

            terminated = infos["terminate"].float()
            terminated = terminated.unsqueeze(-1)
            next_vals = self._eval_critic(self.obs)
            next_vals *= 1.0 - terminated
            self.experience_buffer.update_data("next_values", n, next_vals)

            self.current_rewards += rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            done_indices = all_done_indices[:: self.num_agents]

            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            self.algo_observer.process_infos(infos, done_indices)

            not_dones = 1.0 - self.dones.float()

            if hasattr(self.vec_env.env, "update_motion_weights") and self._prioritized_sampling:
                self._normalized_motion_returns[:] = self.vec_env.env.update_motion_weights(self._normalized_motion_returns, self.current_rewards)
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_next_values = self.experience_buffer.tensor_dict["next_values"]

        mb_rewards = self.experience_buffer.tensor_dict["rewards"]

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        return batch_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["goal_obs"] = batch_dict["goal_obses"]
        return

    def calc_gradients(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        _obs_batch = self._preproc_obs(obs_batch)
        _goal_obs_batch = input_dict["goal_obs"]
        obs_batch = torch.cat([_obs_batch, _goal_obs_batch], dim=-1)

        lr = self.last_lr
        kl = 1.0
        lr_mul = 1.0
        curr_e_clip = lr_mul * self.e_clip

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_len

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info["actor_loss"]

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info["critic_loss"]

            losses, sum_mask = torch_ext.apply_masks([a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1)], rnn_masks)
            a_loss, c_loss, entropy = losses[0], losses[1], losses[2]

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy

            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        # TODO: Refactor this ugliest code of the year
        if self.truncate_grads:
            if self.multi_gpu:
                self.optimizer.synchronize()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                with self.optimizer.skip_synchronize():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

        with torch.no_grad():
            kl_dist = 0.5 * ((old_action_log_probs_batch - action_log_probs) ** 2)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask
            else:
                kl_dist = kl_dist.mean()

        self.train_result = {
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
        }
        self.train_result.update(a_info)
        self.train_result.update(c_info)

        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        self._goal_observation_space = self.env_info["goal_observation_space"]
        self._prioritized_sampling = config.get("prioritized_sampling", False)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config["input_shape"] = (config["input_shape"][-1] + self._goal_observation_space.shape[-1],)
        return config

    def _init_train(self):
        super()._init_train()
        return

    def _build_goal_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["goal_obses"] = torch.zeros(
            batch_shape + self._goal_observation_space.shape, device=self.ppo_device
        )
        self.tensor_list += ["goal_obses"]
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        return
