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
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces
from omegaconf import ListConfig, OmegaConf
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common, schedulers, vecenv
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.a2c_common import rescale_actions
from tensorboardX import SummaryWriter
from torch import nn, optim

import isaacgymenvs.learning.common_discrete_agent as common_discrete_agent
import isaacgymenvs.learning.mlp_discrete_agent as mlp_discrete_agent
import isaacgymenvs.learning.replay_buffer as replay_buffer
from isaacgymenvs.utils.torch_jit_utils import to_torch

class LatentDiscreteAgent(mlp_discrete_agent.MLPDiscreteAgent):
    def __init__(self, base_name, params):
        assert params["load_pretrained"]
        pretrained_path = params["load_pretrained_path"]
        # AMP
        self.use_amp = params.get("use_amp", False)

        # Expert Training
        self.load_pretrained_networks(pretrained_path, params["config"]["device"])

        # setup for low-level cation
        self.clip_actions = params["config"].get('clip_actions', True)

        # bound loss - not using
        self.bounds_loss_coef = params["config"].get("bounds_loss_coef", None)

        super().__init__(base_name, params)

        return

    def preprocess_actions(self, actions):
        if self.clip_actions:
            clamped_actions = torch.clamp(actions, -1.0, 1.0)
            rescaled_actions = rescale_actions(self.actions_low, self.actions_high, clamped_actions)
        else:
            rescaled_actions = actions

        if not self.is_tensor_obses:
            rescaled_actions = rescaled_actions.cpu().numpy()

        return rescaled_actions

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
        if net_config['enc_type'] == "discrete":
            self.pre_model_quantizer = self.pre_model.a2c_network.quantizer
            self.use_continuous_prior = False
        elif net_config['enc_type'] == "hybrid":
            self.pre_model_quantizer = self.pre_model.a2c_network.encoder._post_quantizer
            self.use_continuous_prior = True
        else:
            raise NotImplementedError("invalid type of enc type")

        self.pre_net_config = net_config
        if self.use_amp:
            self._amp_input_mean_std = RunningMeanStd(net_config["amp_input_shape"]).to(device)
            self._amp_input_mean_std.load_state_dict(weights["amp_input_mean_std"])
            self._amp_input_mean_std.eval()
        
        # check if group mode or not
        if hasattr(self.pre_model.a2c_network, "num_groups"):
            self._num_quant_groups = self.pre_model.a2c_network.num_groups
            assert self._num_quant_groups > 1
        else:
            self._num_quant_groups = 1
        return

    def init_tensors(self):
        env_info = copy.deepcopy(self.env_info)
        env_info["action_space"] = self._action_space
        # copy from a2c_common.py (A2CBase)
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            "num_actors": self.num_actors,
            "horizon_length": self.horizon_length,
            "has_central_value": self.has_central_value,
            "use_action_masks": self.use_action_masks,
        }
        self.experience_buffer = ExperienceBuffer(env_info, algo_info, self.ppo_device)

        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        if self.is_rnn:
            self.rnn_states = self.model.get_default_rnn_state()
            self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_len
            assert (self.horizon_length * total_agents // self.num_minibatches) % self.seq_len == 0
            self.mb_rnn_states = [
                torch.zeros(
                    (num_seqs, s.size()[0], total_agents, s.size()[2]), dtype=torch.float32, device=self.ppo_device
                )
                for s in self.rnn_states
            ]

        # copy from a2c_common.py (A2CDiscreteBase)
        self.update_list = ["actions", "neglogpacs", "values"]
        if self.use_action_masks:
            self.update_list += ["action_masks"]
        self.tensor_list = self.update_list + ["obses", "states", "dones"]

        # copy from common_discrete_agent.py
        self.experience_buffer.tensor_dict["next_obses"] = torch.zeros_like(self.experience_buffer.tensor_dict["obses"])
        self.experience_buffer.tensor_dict["next_values"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["values"]
        )

        self.tensor_list += ["next_obses"]

        # copy from mlp_discrete_agent.py
        self._build_goal_buffers()
        if self.use_amp:
            self._build_amp_buffers()
        return

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["amp_obs"] = torch.zeros(
            batch_shape + self._amp_observation_space.shape, device=self.ppo_device
        )
        # self.tensor_list += ["amp_obs"] # we don't have to store the amp observation tensor list
        return

    def _setup_action_space(self):
        code_num = self.pre_net_config["code_num"]
        self.code_num = code_num
        quant_type = self.pre_net_config["quant_type"]
        self.pre_model_quant_type = quant_type
        if quant_type in ["rvq", "rvq_cosine", "rqvq"] or self._num_quant_groups > 1:
            num_quants = self._num_active_quants if not self._use_full_quants else self.pre_net_config["num_quants"]
            assert num_quants <= self.pre_net_config["num_quants"]
            num_quants *= self._num_quant_groups
            self.actions_num = [code_num for _ in range(num_quants)]
            self._action_space = spaces.Tuple([spaces.Discrete(n=code_num) for _ in range(num_quants)])
            self.is_multi_discrete = True
        else:
            self.actions_num = code_num
            self._action_space = spaces.Discrete(n=code_num)
            self.is_multi_discrete = False
        batch_size = self.num_agents * self.num_actors
        self.actions_shape = (self.horizon_length, batch_size)

        # real action dimension in environment
        env_action_space = self.env_info["action_space"]
        self.actions_low = torch.from_numpy(env_action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(env_action_space.high.copy()).float().to(self.ppo_device)
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
            indices = res_dict["actions"] # indices (int64)
            if self._num_quant_groups > 1:
                indices_splits = torch.chunk(indices, self._num_quant_groups, -1)
                latent_action = []
                for i, split in enumerate(indices_splits):
                    latent_action_i = self.pre_model_quantizer[i].get_codes_from_indices(split)
                    if self.pre_model_quant_type in ["rvq", "rfsq"]:
                        latent_action_i = latent_action_i.sum(dim=0)
                    elif self.pre_model_quant_type in ["simple"]:
                        latent_action_i = latent_action_i.squeeze(1)
                    latent_action.append(latent_action_i)
                latent_action = torch.cat(latent_action, dim=-1)
            else:
                latent_action = self.pre_model_quantizer.get_codes_from_indices(indices)
                if self.is_multi_discrete:
                    latent_action = latent_action.sum(dim=0)

            if self.use_continuous_prior: # only for hybrid residual
                if self.pre_net_config.get("post_cond_prior", False):
                    prior_input_dict = {"obs": _processed_obs, "post_z": latent_action}
                else:
                    prior_input_dict = {"obs": _processed_obs}
                prior_dict = self.pre_model.forward_prior(prior_input_dict)
                prior_mu = prior_dict['mu']
                latent_action = latent_action + prior_mu

            pre_model_dict = dict(obs=_processed_obs, latent=latent_action)
            _, res_dict["decoded_actions"] = self.pre_model.decode(**pre_model_dict)
            if self.has_central_value:
                states = obs["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_central_value(input_dict)
                res_dict["values"] = value
        return res_dict

    # AMP setting
    def _preproc_amp_obs(self, amp_obs):
        amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards["disc_rewards"]
        combined_rewards = self._task_reward_w * task_rewards + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.pre_model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {"disc_rewards": disc_r}
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits))
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
            disc_r *= self._disc_reward_scale
        return disc_r

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

            self.obs, rewards, self.dones, infos = self.env_step(res_dict["decoded_actions"])
            shaped_rewards = self.rewards_shaper(rewards)
            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.experience_buffer.update_data("next_obses", n, self.obs["obs"])
            self.experience_buffer.update_data("dones", n, self.dones)
            if self.use_amp:
                self.experience_buffer.update_data("amp_obs", n, infos["amp_obs"])

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
        if self.use_amp:
            mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]
            amp_rewards = self._calc_amp_rewards(mb_amp_obs)
            mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

        mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size

        return batch_dict

    def train_epoch(self):
        play_time_start = time.time()
        with torch.no_grad():
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()

        play_time_end = time.time()
        update_time_start = time.time()
        rnn_masks = batch_dict.get("rnn_masks", None)

        self.set_train()

        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        if self.has_central_value:
            self.train_central_value()

        train_info = None

        if self.is_rnn:
            frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
            print(frames_mask_ratio)

        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                curr_train_info = self.train_actor_critic(self.dataset[i])

                if self.schedule_type == "legacy":
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr,
                        self.entropy_coef,
                        self.epoch_num,
                        0,
                        curr_train_info["kl"].item(),
                    )
                    self.update_lr(self.last_lr)

                if train_info is None:
                    train_info = dict()
                    for k, v in curr_train_info.items():
                        train_info[k] = [v]
                else:
                    for k, v in curr_train_info.items():
                        train_info[k].append(v)

            av_kls = torch_ext.mean_list(train_info["kl"])

            if self.schedule_type == "standard":
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                )
                self.update_lr(self.last_lr)

        if self.schedule_type == "standard_epoch":
            self.last_lr, self.entropy_coef = self.scheduler.update(
                self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
            )
            self.update_lr(self.last_lr)

        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        train_info["play_time"] = play_time
        train_info["update_time"] = update_time
        train_info["total_time"] = total_time
        self._record_train_batch_info(batch_dict, train_info)

        return train_info

    def calc_gradients(self, input_dict):
        self.train_result = dict()
        self.calc_gradients_rl(input_dict)
        return

    def calc_gradients_rl(self, input_dict):
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

        rl_info = {
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
        }
        self.train_result.update(rl_info)
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        self._goal_observation_space = self.env_info["goal_observation_space"]
        self._prioritized_sampling = config.get("prioritized_sampling", False)
        self._num_active_quants = config.get("num_quants", -1)
        self._use_full_quants = True if self._num_active_quants == -1 else False
        if self.use_amp:
            self._amp_observation_space = self.env_info["amp_observation_space"]
            self._task_reward_w = config["task_reward_w"]
            self._disc_reward_w = config["disc_reward_w"]
            self._disc_reward_scale = config["disc_reward_scale"]
        return

    def _init_train(self):
        super()._init_train()
        return

    def bound_loss(self, mu):
        if self.bounds_loss_coef is not None:
            soft_bound = 1.0
            mu_loss_high = torch.maximum(mu - soft_bound, torch.tensor(0, device=self.ppo_device)) ** 2
            mu_loss_low = torch.minimum(mu + soft_bound, torch.tensor(0, device=self.ppo_device)) ** 2
            b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
        else:
            b_loss = 0
        return b_loss