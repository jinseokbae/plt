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
from omegaconf import ListConfig, OmegaConf
from pathlib import Path

import numpy as np
import torch
from rl_games.algos_torch import torch_ext, model_builder
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common, schedulers, vecenv
from tensorboardX import SummaryWriter
from torch import nn, optim

import isaacgymenvs.learning.mlp_continuous_agent as mlp_continuous_agent
import isaacgymenvs.learning.replay_buffer as replay_buffer
from isaacgymenvs.utils.torch_jit_utils import to_torch

from rl_games.common.experience import ExperienceBuffer
from gym import spaces


class LatentContinuousAgent(mlp_continuous_agent.MLPContinuousAgent):
    def __init__(self, base_name, params):
        assert params["load_pretrained"]
        pretrained_path = params["load_pretrained_path"]

        # AMP
        self.use_amp = params.get("use_amp", False)

        # Expert Training
        self.use_expert = params.get("use_latent_expert", False)
        self.load_pretrained_networks(pretrained_path, params["config"]["device"])
        self.distill = True if self.use_expert else False

        # Exploration
        self.decoder_noise = params["config"].get("decoder_noise", True)

        super().__init__(base_name, params)
        if self.use_expert:
            actor_lr = params["config"].get("actor_learning_rate", float(self.last_lr))
            param_name_filter = ['actor_mlp', 'mu', 'sigma']
            def _name_filter(name, qs):
                for q in qs:
                    if q in name:
                        return True
                return False
            actor_parameters = [param for name, param in self.model.named_parameters() if _name_filter(name, param_name_filter)]
            self.optimizer = optim.Adam(
                actor_parameters,
                actor_lr,
                eps=1e-08,
                weight_decay=self.weight_decay,
            )
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
        if self.use_expert:
            expert_goal_observation_shape = weights["goal_input_mean_std"]["running_mean"].shape
            expert_goal_input_mean_std = RunningMeanStd(expert_goal_observation_shape)
            expert_goal_input_mean_std.load_state_dict(weights["goal_input_mean_std"])
            self._expert_goal_input_mean_std = expert_goal_input_mean_std
            self._expert_goal_input_mean_std.eval()
            self._expert_goal_input_mean_std = self._expert_goal_input_mean_std.to(device)
        if self.use_amp:
            self._amp_input_mean_std = RunningMeanStd(net_config["amp_input_shape"]).to(device)
            self._amp_input_mean_std.load_state_dict(weights["amp_input_mean_std"])
            self._amp_input_mean_std.eval()
        if self.pre_net_config["enc_type"] == "continuous":
            self.use_continuous_prior = True if self.pre_net_config["continuous_enc_style"] == "statecond" else False
            self.posterior_quantize = False
        elif self.pre_net_config["enc_type"] == "hybrid":
            self.use_continuous_prior = True
            self.posterior_quantize = False
        else:
            self.use_continuous_prior = False
            self.posterior_quantize = False
        return

    def init_tensors(self):
        super().init_tensors()  # already built goal buffers
        # redefine action dimension in experence_buffer
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.action_space = self._action_space
        self.experience_buffer.actions_shape = (self.experience_buffer.action_space.shape[0],)
        self.experience_buffer.actions_num = self.experience_buffer.action_space.shape[0]
        self.experience_buffer.tensor_dict["actions"] = torch.zeros(
            batch_shape + self._action_space.shape, device=self.ppo_device
        )
        self.experience_buffer.tensor_dict["mus"] = torch.zeros_like(self.experience_buffer.tensor_dict["actions"])
        self.experience_buffer.tensor_dict["sigmas"] = torch.zeros_like(self.experience_buffer.tensor_dict["actions"])
        if self.use_expert:
            self._build_expert_latent_buffers()
        if self.use_amp:
            self._build_amp_buffers()
    
    def _build_expert_latent_buffers(self):
        self.experience_buffer.tensor_dict["expert_latents"] = torch.zeros_like(self.experience_buffer.tensor_dict["mus"])
        self.tensor_list += ["expert_latents"]
        return

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["amp_obs"] = torch.zeros(
            batch_shape + self._amp_observation_space.shape, device=self.ppo_device
        )
        # self.tensor_list += ["amp_obs"] # we don't have to store the amp observation tensor list
        return

    def _setup_action_space(self):
        self.actions_num = self.pre_net_config["latent_shape"][-1]
        self._action_space = spaces.Box(np.ones(self.actions_num) * -np.inf, np.ones(self.actions_num) * np.inf)

        # real action dimension in environment
        env_action_space = self.env_info["action_space"]
        self.actions_low = torch.from_numpy(env_action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(env_action_space.high.copy()).float().to(self.ppo_device)
        return

    def get_action_values(self, obs):
        _processed_obs = self._preproc_obs(obs["obs"])
        _processed_goal_obs = obs["goal_obs"]
        processed_obs = torch.cat([_processed_obs, _processed_goal_obs], dim=-1)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": processed_obs,
            "rnn_states": self.rnn_states,
        }
        self.model.eval()
        with torch.no_grad():
            # forward
            res_dict = self.model(input_dict)
            latent_mu = res_dict["mus"]
            latent_action = res_dict["actions"]

            if self.posterior_quantize:
                quant_mu, *_ = self.pre_model.a2c_network.encoder._post_quantizer(latent_mu[:, None])
                latent_mu = quant_mu[:, 0] # no graident
            if self.use_continuous_prior:
                prior_input_dict = {"obs": _processed_obs}
                prior_dict = self.pre_model.forward_prior(prior_input_dict)
                prior_mu = prior_dict['mu']
                latent_mu = latent_mu + prior_mu
                latent_action = latent_action + prior_mu

            # add noise
            if self.decoder_noise or self.distill:
                latent = latent_mu
            else:
                latent = latent_action

            pre_model_dict = dict(obs=_processed_obs, latent=latent)
            
            if self.decoder_noise:
                res_dict["decoded_actions"], _ = self.pre_model.decode(**pre_model_dict) # use stochastic action for exploration
            else:
                _, res_dict["decoded_actions"] = self.pre_model.decode(**pre_model_dict) # use deterministic action, but exploration in the latent space

            if self.has_central_value:
                states = obs["states"]
                input_dict = {
                    "is_train": False,
                    "states": states,
                }
                value = self.get_central_value(input_dict)
                res_dict["values"] = value
        return res_dict

    def get_expert_goal_obs(self):
        with torch.no_grad():
            expert_goal_obs = self.vec_env.env._compute_ref_diff_observations()
        return expert_goal_obs

    def get_expert_latent_values(self, obs):
        processed_obs = self._preproc_obs(obs["obs"])
        processed_goal_obs = self._expert_goal_input_mean_std(obs["goal_obs"])
        input_dict = {
            "obs": processed_obs,
            "goal_obs": processed_goal_obs,
        }

        with torch.no_grad():
            res_dict = self.pre_model.forward_latent(input_dict)
        return res_dict["latents"]

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

            # expert
            if self.use_expert:
                expert_goal_obs = self.get_expert_goal_obs()
                expert_obs = dict(obs=self.obs["obs"], goal_obs=expert_goal_obs)
                expert_latents = self.get_expert_latent_values(expert_obs)
                self.experience_buffer.update_data("expert_latents", n, expert_latents)
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

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        if self.use_expert:
            self.dataset.values_dict["expert_latent"] = batch_dict["expert_latents"]
    
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

                if self.schedule_type == "legacy" and not self.distill:
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

            if not self.distill:
                av_kls = torch_ext.mean_list(train_info["kl"])

                if self.schedule_type == "standard":
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
                    )
                    self.update_lr(self.last_lr)

        if self.schedule_type == "standard_epoch" and not self.distill:
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
        if self.distill:
            self.calc_gradients_non_rl(input_dict)
        else:
            self.calc_gradients_rl(input_dict)
        return

    def calc_gradients_non_rl(self, input_dict):
        self.set_train()
        obs_batch = input_dict["obs"]
        _obs_batch = self._preproc_obs(obs_batch)
        _goal_obs_batch = input_dict["goal_obs"]
        obs_batch = torch.cat([_obs_batch, _goal_obs_batch], dim=-1)
        expert_latent_batch = input_dict["expert_latent"]

        actions_batch = input_dict["actions"]

        batch_dict = {
            "is_train": True,
            "prev_actions": actions_batch,
            "obs": obs_batch,
        }

        # whether using prior or not
        if self.use_continuous_prior:
            prior_input_dict = {"obs": _obs_batch}
            prior_dict = self.pre_model.forward_prior(prior_input_dict)
            prior_mu = prior_dict['mu']
        else:
            prior_mu = None

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # forward
            res_dict = self.model(batch_dict)
            latent_mu = res_dict["mus"]

            if self.posterior_quantize:
                quant_mu, *_ = self.pre_model.a2c_network.encoder._post_quantizer(latent_mu[:, None])
                quant_mu = quant_mu[:, 0] # no graident
                latent_mu = latent_mu + (quant_mu - latent_mu).detach() # straight through
            if self.use_continuous_prior:
                latent_mu = latent_mu + prior_mu

            # expert loss
            latent = latent_mu
            expert_loss = (latent - expert_latent_batch).pow(2).sum(dim=-1).mean()
            loss = expert_loss

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

        train_result = {
            "expert_loss": expert_loss
        }
        self.train_result.update(train_result)
        return

    def calc_gradients_rl(self, input_dict):
        self.set_train()

        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
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

        # whether using prior or not
        if self.use_continuous_prior:
            prior_input_dict = {"obs": _obs_batch}
            prior_dict = self.pre_model.forward_prior(prior_input_dict)
            prior_mu = prior_dict['mu']
        else:
            prior_mu = None

        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            # forward
            res_dict = self.model(batch_dict)
            latent_mu = res_dict["mus"]

            if self.posterior_quantize:
                quant_mu, *_ = self.pre_model.a2c_network.encoder._post_quantizer(latent_mu[:, None])
                quant_mu = quant_mu[:, 0] # no graident
                latent_mu = latent_mu + (quant_mu - latent_mu).detach() # straight through
            if self.use_continuous_prior:
                latent_mu = latent_mu + prior_mu

            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = latent_mu
            sigma = res_dict["sigmas"]

            # decoded action
            pre_model_dict = dict(obs=_obs_batch, latent=mu)
            _, decoded_mus = self.pre_model.decode(**pre_model_dict) # use deterministic action

            a_info = self._actor_loss(old_action_log_probs_batch, action_log_probs, advantage, curr_e_clip)
            a_loss = a_info["actor_loss"]

            c_info = self._critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            c_loss = c_info["critic_loss"]

            b_loss = self.bound_loss(decoded_mus)

            losses, sum_mask = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss, entropy.unsqueeze(1), b_loss.unsqueeze(1)],
                rnn_masks,
            )
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]

            loss = a_loss + self.critic_coef * c_loss - self.entropy_coef * entropy + self.bounds_loss_coef * b_loss

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
            reduce_kl = not self.is_rnn
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if self.is_rnn:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  # / sum_mask

        rl_info = {
            "entropy": entropy,
            "kl": kl_dist,
            "last_lr": self.last_lr,
            "lr_mul": lr_mul,
            "b_loss": b_loss,
        }
        self.train_result.update(rl_info)
        self.train_result.update(a_info)
        self.train_result.update(c_info)
        return

    def _load_config_params(self, config):
        super()._load_config_params(config)
        self._goal_observation_space = self.env_info["goal_observation_space"]
        self._prioritized_sampling = config.get("prioritized_sampling", False)
        if self.use_amp:
            self._amp_observation_space = self.env_info["amp_observation_space"]
            self._task_reward_w = config["task_reward_w"]
            self._disc_reward_w = config["disc_reward_w"]
            self._disc_reward_scale = config["disc_reward_scale"]
        return

    def _init_train(self):
        super()._init_train()
        return

    def _log_train_info(self, train_info, frame):
        if self.distill:
            self.writer.add_scalar(
                "losses/expert_loss",
                torch_ext.mean_list(train_info["expert_loss"]).item(),
                frame,
            )
        else:
            super()._log_train_info(train_info, frame)

        return