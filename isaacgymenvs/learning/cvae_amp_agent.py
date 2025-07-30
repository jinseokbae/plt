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

import time

import numpy as np
import torch
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from torch import nn

import isaacgymenvs.learning.replay_buffer as replay_buffer
from isaacgymenvs.utils.torch_jit_utils import to_torch
import isaacgymenvs.learning.cvae_agent as cvae_agent


class cVAEAMPAgent(cvae_agent.cVAEAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)

        return

    def init_tensors(self):
        super().init_tensors()
        self._build_amp_buffers()
        return

    def set_eval(self):
        super().set_eval()
        if self._normalize_amp_input:
            self._amp_input_mean_std.eval()
        return

    def set_train(self):
        super().set_train()
        if self._normalize_amp_input:
            self._amp_input_mean_std.train()
        return

    def get_stats_weights(self):
        state = super().get_stats_weights()
        if self._normalize_amp_input:
            state["amp_input_mean_std"] = self._amp_input_mean_std.state_dict()
        return state

    def set_stats_weights(self, weights):
        super().set_stats_weights(weights)
        if self._normalize_amp_input:
            self._amp_input_mean_std.load_state_dict(weights["amp_input_mean_std"])
        return

    def play_steps_maybe_update_amp_obs(self, n, infos):
        self.experience_buffer.update_data("amp_obs", n, infos["amp_obs"])
        return
    
    def play_steps_maybe_amp_debug(self, n, infos):
        if self.vec_env.env.viewer and (n == (self.horizon_length - 1)):
            self._amp_debug(infos)
        return

    def play_steps_maybe_add_amp_reward(self, mb_rewards):
        mb_amp_obs = self.experience_buffer.tensor_dict["amp_obs"]
        amp_rewards = self._calc_amp_rewards(mb_amp_obs)
        mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)
        return mb_rewards, amp_rewards

    def play_steps_maybe_swap_amp_rewards(self, amp_rewards, batch_dict):
        for k, v in amp_rewards.items():
            batch_dict[k] = a2c_common.swap_and_flatten01(v)
        return batch_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict["amp_obs"] = batch_dict["amp_obs"]
        self.dataset.values_dict["amp_obs_demo"] = batch_dict["amp_obs_demo"]
        self.dataset.values_dict["amp_obs_replay"] = batch_dict["amp_obs_replay"]
        return

    def train_epoch_maybe_update_amp_buffers(self, batch_dict):
        self._update_amp_demos()
        num_obs_samples = batch_dict["amp_obs"].shape[0]
        amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)["amp_obs"]
        batch_dict["amp_obs_demo"] = amp_obs_demo

        if self._amp_replay_buffer.get_total_count() == 0:
            batch_dict["amp_obs_replay"] = batch_dict["amp_obs"]
        else:
            batch_dict["amp_obs_replay"] = self._amp_replay_buffer.sample(num_obs_samples)["amp_obs"]
        return batch_dict

    def train_epoch_maybe_store_replay_amp_obs(self, batch_dict):
        if not self.distill:
            self._store_replay_amp_obs(batch_dict["amp_obs"])
        return

    def calc_gradients_rl_maybe_prepare_amp_obs(self, input_dict, batch_dict):
        # AMP
        amp_obs = input_dict["amp_obs"][0 : self._amp_minibatch_size]
        amp_obs = self._preproc_amp_obs(amp_obs)
        amp_obs_replay = input_dict["amp_obs_replay"][0 : self._amp_minibatch_size]
        amp_obs_replay = self._preproc_amp_obs(amp_obs_replay)

        amp_obs_demo = input_dict["amp_obs_demo"][0 : self._amp_minibatch_size]
        amp_obs_demo = self._preproc_amp_obs(amp_obs_demo)
        amp_obs_demo.requires_grad_(True)
        amp_batch_dict = {
            "amp_obs": amp_obs,
            "amp_obs_replay": amp_obs_replay,
            "amp_obs_demo": amp_obs_demo,
        }
        batch_dict.update(amp_batch_dict)
        return batch_dict


    def calc_gradients_rl_maybe_disc_loss(self, res_dict, batch_dict):
        amp_obs_demo = batch_dict["amp_obs_demo"]
        disc_agent_logit = res_dict["disc_agent_logit"]
        disc_agent_replay_logit = res_dict["disc_agent_replay_logit"]
        disc_demo_logit = res_dict["disc_demo_logit"]
        disc_agent_cat_logit = torch.cat([disc_agent_logit, disc_agent_replay_logit], dim=0)
        disc_info = self._disc_loss(disc_agent_cat_logit, disc_demo_logit, amp_obs_demo)
        disc_loss = self._disc_coef * disc_info["disc_loss"]
        self.train_result.update(disc_info)
        return disc_loss


    def _load_config_params(self, config):
        super()._load_config_params(config)

        self._task_reward_w = config["task_reward_w"]
        self._disc_reward_w = config["disc_reward_w"]

        self._amp_observation_space = self.env_info["amp_observation_space"]
        self._amp_batch_size = int(config["amp_batch_size"])
        self._amp_minibatch_size = int(config["amp_minibatch_size"])
        assert self._amp_minibatch_size <= self.minibatch_size

        self._disc_coef = config["disc_coef"]
        self._disc_logit_reg = config["disc_logit_reg"]
        self._disc_grad_penalty = config["disc_grad_penalty"]
        self._disc_weight_decay = config["disc_weight_decay"]
        self._disc_reward_scale = config["disc_reward_scale"]
        self._normalize_amp_input = config.get("normalize_amp_input", True)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        config["amp_input_shape"] = self._amp_observation_space.shape
        return config

    def _init_train(self):
        super()._init_train()
        self._init_amp_demo_buf()
        return

    def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
        # prediction loss
        disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
        disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
        disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

        # logit reg
        logit_weights = self.model.a2c_network.get_disc_logit_weights()
        disc_logit_loss = torch.sum(torch.square(logit_weights))
        disc_loss += self._disc_logit_reg * disc_logit_loss

        # grad penalty
        disc_demo_grad = torch.autograd.grad(
            disc_demo_logit,
            obs_demo,
            grad_outputs=torch.ones_like(disc_demo_logit),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        disc_demo_grad = disc_demo_grad[0]
        disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
        disc_grad_penalty = torch.mean(disc_demo_grad)
        disc_loss += self._disc_grad_penalty * disc_grad_penalty

        # weight decay
        if self._disc_weight_decay != 0:
            disc_weights = self.model.a2c_network.get_disc_weights()
            disc_weights = torch.cat(disc_weights, dim=-1)
            disc_weight_decay = torch.sum(torch.square(disc_weights))
            disc_loss += self._disc_weight_decay * disc_weight_decay

        disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

        disc_info = {
            "disc_loss": disc_loss,
            "disc_grad_penalty": disc_grad_penalty,
            "disc_logit_loss": disc_logit_loss,
            "disc_agent_acc": disc_agent_acc,
            "disc_demo_acc": disc_demo_acc,
            "disc_agent_logit": disc_agent_logit,
            "disc_demo_logit": disc_demo_logit,
        }
        return disc_info

    def _disc_loss_neg(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.zeros_like(disc_logits))
        return loss

    def _disc_loss_pos(self, disc_logits):
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(disc_logits, torch.ones_like(disc_logits))
        return loss

    def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
        agent_acc = disc_agent_logit < 0
        agent_acc = torch.mean(agent_acc.float())
        demo_acc = disc_demo_logit > 0
        demo_acc = torch.mean(demo_acc.float())
        return agent_acc, demo_acc

    def _fetch_amp_obs_demo(self, num_samples):
        amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
        return amp_obs_demo

    def _build_amp_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        self.experience_buffer.tensor_dict["amp_obs"] = torch.zeros(
            batch_shape + self._amp_observation_space.shape, device=self.ppo_device
        )

        amp_obs_demo_buffer_size = int(self.config["amp_obs_demo_buffer_size"])
        self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        self._amp_replay_keep_prob = self.config["amp_replay_keep_prob"]
        replay_buffer_size = int(self.config["amp_replay_buffer_size"])
        self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ["amp_obs"]
        return

    def _init_amp_demo_buf(self):
        buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
        num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

        for i in range(num_batches):
            curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
            self._amp_obs_demo_buffer.store({"amp_obs": curr_samples})

        return

    def _update_amp_demos(self):
        new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
        self._amp_obs_demo_buffer.store({"amp_obs": new_amp_obs_demo})
        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _combine_rewards(self, task_rewards, amp_rewards):
        disc_r = amp_rewards["disc_rewards"]
        combined_rewards = self._task_reward_w * task_rewards + self._disc_reward_w * disc_r
        return combined_rewards

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

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

    def _store_replay_amp_obs(self, amp_obs):
        buf_size = self._amp_replay_buffer.get_buffer_size()
        buf_total_count = self._amp_replay_buffer.get_total_count()
        if buf_total_count > buf_size:
            keep_probs = to_torch(
                np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]),
                device=self.ppo_device,
            )
            keep_mask = torch.bernoulli(keep_probs) == 1.0
            amp_obs = amp_obs[keep_mask]

        self._amp_replay_buffer.store({"amp_obs": amp_obs})
        return

    def _record_train_batch_info(self, batch_dict, train_info):
        train_info["disc_rewards"] = batch_dict["disc_rewards"]
        return

    def _log_train_info(self, train_info, frame):
        super()._log_train_info(train_info, frame)
        if not self.distill:
            self.writer.add_scalar(
                "losses/disc_loss",
                torch_ext.mean_list(train_info["disc_loss"]).item(),
                frame,
            )

            self.writer.add_scalar(
                "info/disc_agent_acc",
                torch_ext.mean_list(train_info["disc_agent_acc"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_demo_acc",
                torch_ext.mean_list(train_info["disc_demo_acc"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_agent_logit",
                torch_ext.mean_list(train_info["disc_agent_logit"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_demo_logit",
                torch_ext.mean_list(train_info["disc_demo_logit"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_grad_penalty",
                torch_ext.mean_list(train_info["disc_grad_penalty"]).item(),
                frame,
            )
            self.writer.add_scalar(
                "info/disc_logit_loss",
                torch_ext.mean_list(train_info["disc_logit_loss"]).item(),
                frame,
            )

            disc_reward_std, disc_reward_mean = torch.std_mean(train_info["disc_rewards"])
            self.writer.add_scalar("info/disc_reward_mean", disc_reward_mean.item(), frame)
            self.writer.add_scalar("info/disc_reward_std", disc_reward_std.item(), frame)
        return

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info["amp_obs"]
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards["disc_rewards"]

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)
        return
