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
import time
import os
import json

import isaacgymenvs.learning.common_player as common_player

import pathlib # (dglim - visualization)
from torch.distributions.kl import kl_divergence
from torch.distributions.categorical import Categorical
PRIOR_ROLLOUT_MAX_NSAMPLE = 800000

class cVAEPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, params):
        # Basic config processing
        config = params["config"]
        self.random_seed = params["seed"] # for naming
        self._normalize_goal_input = config.get("normalize_goal_input", True)
        self._no_goal = config.get("no_goal", False) # (Aborted) ablation
        self._latent_dim = config.get("latent_dim", 64) if not self._no_goal else 0
        self._enc_type = config.get("enc_type", "continuous")
        if self._enc_type in ["continuous", "hybrid"]:
            self._enc_scale = config.get("enc_scale", 0.3)
            self._continuous_enc_style = config.get("continuous_enc_style", "standard")
        if self._enc_type in ["discrete", "hybrid"]:
            self._code_num = config.get("code_num", 512)
            self._quant_type = config.get("quant_type", "basic")
            self._num_quants = config.get("num_quants", 4)
            self._num_inference_quants = config.get("num_inference_quants", 0)
        if self._enc_type == "hybrid":
            self._post_cond_prior = config.get("post_cond_prior", False)
        
        self._dof_group = config.get("dof_group", None)

        super().__init__(params)

        self.prior_rollout = self.env.cfg["env"].get("prior_rollout", False)
        return

    def restore(self, fn):
        super().restore(fn)
        if self._normalize_goal_input:
            checkpoint = torch_ext.load_checkpoint(fn)
            self._goal_input_mean_std.load_state_dict(checkpoint["goal_input_mean_std"])

        # (dglim - visualization)
        self.checkpoint_fn = pathlib.Path(fn).stem
        self.env.checkpoint_fn = self.checkpoint_fn
        
        # motion matching - eval prior rollout quantitatitvely
        if self.env.cfg["env"]["motion_matching"]:
            split_fn = fn.split('/')
            assert split_fn[-2] == 'nn'
            coverage_save_dir = '/'.join(split_fn[:-2])
            num_envs = self.env.num_envs
            epi_len = self.env.max_episode_length
            n_epoch = checkpoint['epoch']
            self.coverage_save_file = os.path.join(coverage_save_dir, f'prior_rollout_seed{self.random_seed}_env{num_envs}_epilen{epi_len}_epoch{n_epoch}.json')
            self.coverage_data = dict()

        # using portion of codebooks
        if self._enc_type in ["discrete", "hybrid"]:
            if self._quant_type == "rvq":
                if self._num_inference_quants > 0:
                    if self._enc_type == "discrete":
                        quantizer = self.model.a2c_network.quantizer
                    else:
                        quantizer = self.model.a2c_network.encoder._post_quantizer
                    quantizer.layers = quantizer.layers[:self._num_inference_quants]
        return

    def _build_net(self, config):
        super()._build_net(config)

        if self._normalize_goal_input:
            self._goal_input_mean_std = RunningMeanStd(config["goal_input_shape"]).to(self.device)
            self._goal_input_mean_std.eval()
        return

    def _post_step(self, info):
        super()._post_step(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if hasattr(self, "env"):
            config["goal_input_shape"] = self.env.goal_observation_space.shape
        else:
            config["goal_input_shape"] = self.env_info["goal_observation_space"]

        config["no_goal"] = self._no_goal
        config["latent_shape"] = (self._latent_dim,)
        print("Latent Dimension : ", self._latent_dim)
        config["enc_type"] = self._enc_type
        if self._enc_type in ["continuous", "hybrid"]:
            config["enc_scale"] = self._enc_scale
            config["continuous_enc_style"] = self._continuous_enc_style
        if self._enc_type in ["discrete", "hybrid"]:
            config["code_num"] = self._code_num
            config["quant_type"] = self._quant_type
            config["num_quants"] = self._num_quants
        if self._enc_type == "hybrid":
            config["post_cond_prior"] = self._post_cond_prior

        if self._dof_group != None:
            if self._dof_group == 'upper-lower':
                config["dof_group"] = [
                    [i for i in range(14)], # upper
                    [i for i in range(14,28)], # lower
                ]
            elif self._dof_group == 'right-left':
                config["dof_group"] = [
                    [0, 1, 2, 3, 4, 5] + [ 6,  7,  8,  9] + [14, 15, 16, 17, 18, 19, 20], # trunk + right arm + right leg
                    [0, 1, 2, 3, 4, 5] + [10, 11, 12, 13] + [21, 22, 23, 24, 25, 26, 27], # trunk + left arm + left leg
                ]
            elif self._dof_group == 'trunk-limbs':
                config["dof_group"] = [
                    [i for i in range(6)], # trunk - abdomen + neck
                    [i for i in range(6,10)], # limb1 - right arm
                    [i for i in range(10,14)], # limb2 - left arm
                    [i for i in range(14,21)], # limb3 - right leg
                    [i for i in range(21,28)], # limb4 - left leg
                ]
            elif self._dof_group == 'random5':
                config["dof_group"] = [
                    [14, 15, 7, 22, 3, 10], 
                    [2, 5, 17, 12, 26, 6], 
                    [0, 24, 21, 16, 9, 18],
                    [23, 20, 13, 1, 4, 19], 
                    [8, 25, 27, 11], 
                ]
            _dof_id_sets = []
            for group in config["dof_group"]:
                _dof_id_sets += group
            _dof_id_sets = list(set(_dof_id_sets))
            assert _dof_id_sets == [i for i in range(config["actions_num"])] # assert if all the dof ids are included in groups

        return config

    def _preproc_goal_obs(self, goal_obs):
        if self._normalize_goal_input:
            goal_obs = self._goal_input_mean_std(goal_obs)
        return goal_obs

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

    # most important function in this file, policy inferring action
    # by default, is_deterministic=True for smooth trajectory collection
    def get_action(self, obs, is_deterministic=False):
        goal_obs = obs["goal_obs"]
        obs = obs["obs"]

        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
            goal_obs = unsqueeze_obs(goal_obs)

        obs = self._preproc_obs(obs)
        goal_obs = self._preproc_goal_obs(goal_obs)
        input_dict = {
            "is_train": False,
            "prev_actions": None,
            "obs": obs,
            "goal_obs": goal_obs,
            "rnn_states": self.states,
        }
        with torch.no_grad():
            if self.prior_rollout:
                if hasattr(self, "_num_inference_quants"):
                    if self._num_inference_quants > 0:
                        input_dict["num_active_quants"] = self._num_inference_quants
                res_dict = self.model.prior_rollout(input_dict)
            else:
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

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        need_init_rnn = self.is_rnn


        try:
            self.model.a2c_network.generate_multihead_quantizer()
        except:
            pass

        avg_durs = []
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict["obs"], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            if self.env.cfg["env"]["motion_matching"]:
                self.max_steps = int(1e12)

            dur = []
            for n in range(self.max_steps):
                self.local_step = n
                obs_dict, done_env_ids = self._env_reset_done()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    start = time.time()
                    action = self.get_action(obs_dict, is_determenistic)
                    if n < 10:
                        dur.append(time.time() - start)
                obs_dict, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1

                self._post_step(info)

                if render:
                    self.env.render(mode="human")
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[:: self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    avg_durs.append(sum(dur) / len(dur))
                    if games_played == 10:
                        print(sum(avg_durs) / len(avg_durs), "Seconds")
                    self.maybe_capture_video()
                    if self.eval_jitter:
                        jitter = self.jitter / self.env.max_episode_length
                        jerk = self.jerk / self.env.max_episode_length
                        print("Average Jitter: %.4f (%.4f)" % (jitter.mean().item(), jitter.std().item()))
                        print("Average Jerk: %.2f (%.3f)" % (jerk.mean().item(), jerk.std().item()))
                    if hasattr(self.env, "metric"):
                        # tracking metric
                        mpjpe = self.env.metric["mpjpe_l"][self.env.num_envs:].mean(axis=1)
                        gmpjpe = self.env.metric["mpjpe_g"][self.env.num_envs:].mean(axis=1)
                        vel_error = self.env.metric["vel_dist"][self.env.num_envs:]
                        accel_error = self.env.metric["accel_dist"][self.env.num_envs:]
                        print("Average MPJPE: %.3f (%.3f)" % (mpjpe.mean().item(), mpjpe.std().item()))
                        print("Average G-MPJPE: %.3f (%.3f)" % (gmpjpe.mean().item(), gmpjpe.std().item()))
                        print("Average VEL-ERROR: %.3f (%.3f)" % (vel_error.mean().item(), vel_error.std().item()))
                        print("Average ACCEL-ERROR: %.3f (%.3f)" % (accel_error.mean().item(), accel_error.std().item()))
                        if "success_rate" in self.env.metric:
                            success_rate = self.env.metric["success_rate"]
                            print("Success Rate: %.4f" % (success_rate * 100))
                    if self.env.cfg["env"]["motion_matching"] and hasattr(self.env, "matched_counts"):
                        valid_samples = self.env.matched_counts.sum().item()
                        n_samples = self.env.total_samples
                        filtering_rate = (1 - valid_samples / n_samples) * 100
                        print()
                        print("Total Samples: ", n_samples)
                        print("Filtering Rate: ", filtering_rate)

                        matching_distance = self.env.mean_matched_dist / self.env.max_episode_length
                        self.env.mean_matched_dist = 0 # reset
                        print(f"Mean Matching Distance: {matching_distance: .2f}")

                        coverage = (self.env.matched_counts > 0).sum().item() / self.env.matched_counts.shape[0] * 100
                        print(f"Coverage :{coverage: .2f} %")

                        self.coverage_data[n_samples] = dict(
                            dist=matching_distance,
                            coverage=coverage,
                            filtering_rate=filtering_rate,
                        )
                        with open(self.coverage_save_file, 'w') as f:
                            json.dump(self.coverage_data, f)
                        if n_samples > PRIOR_ROLLOUT_MAX_NSAMPLE:
                            return
                        
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:, all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_reward_std = cr[done_indices].std().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if "battle_won" in info:
                            print_game_res = True
                            game_res = info.get("battle_won", 0.5)
                        if "scores" in info:
                            print_game_res = True
                            game_res = info.get("scores", 0.5)
                    if self.print_stats:
                        if print_game_res:
                            print(
                                "reward:",
                                cur_rewards / done_count,
                                "steps:",
                                cur_steps / done_count,
                                "w:",
                                game_res,
                            )
                        else:
                            print(
                                "reward: %.3f (%.3f)" % (cur_rewards / done_count, cur_reward_std),
                                "steps:",
                                cur_steps / done_count,
                            )

                    sum_game_res += game_res
                    if batch_size // self.num_agents == 1 or games_played >= n_games:
                        break

        print(sum_rewards)
        if print_game_res:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
                "winrate:",
                sum_game_res / games_played * n_game_life,
            )
        else:
            print(
                "av reward:",
                sum_rewards / games_played * n_game_life,
                "av steps:",
                sum_steps / games_played * n_game_life,
            )

        return