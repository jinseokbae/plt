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
import sys
from rl_games.algos_torch import players, torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer


EXTRACT_BINS = False
class CommonPlayer(players.PpoPlayerContinuous):
    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config["network"]

        self.normalize_input = self.config["normalize_input"]
        self.normalize_value = self.config["normalize_value"]

        self._setup_action_space()
        self.mask = [False]

        net_config = self._build_net_config()
        self._build_net(net_config)
        
        # jsbae
        self.eval_jitter = params["config"].get("eval_jitter", False)
        return

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

            for n in range(self.max_steps):
                self.local_step = n
                obs_dict, done_env_ids = self._env_reset_done()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
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
                    self.maybe_capture_video()
                    if self.eval_jitter:
                        jitter = self.jitter / self.env.max_episode_length
                        jerk = self.jerk / self.env.max_episode_length
                        print("Average Jitter: %.4f (%.4f)" % (jitter.mean().item(), jitter.std().item()))
                        print("Average Jerk: %.2f (%.2f)" % (jerk.mean().item(), jerk.std().item()))
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
                            print("Success Rate: %.2f" % (success_rate * 100))

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

        if EXTRACT_BINS: # for latent visualization
            import numpy as np
            bins = self.model.a2c_network._bins.numpy().astype(np.int32)
        return

    def obs_to_torch(self, obs):
        upd_obs = super().obs_to_torch(obs)
        obs_dict = {"obs": upd_obs}
        return obs_dict

    def get_action(self, obs_dict, is_determenistic=False):
        output = super().get_action(obs_dict["obs"], is_determenistic)
        return output

    def _build_net(self, config):
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

        return

    def _collect_jitter_info(self, done_env_ids):
        if self.eval_jitter:
            assert not self.env._enable_early_termination # only valid for not early termination
            if done_env_ids.shape[0] > 0: # first frame
                if not hasattr(self, "_prev_humanoid_dof_vel"):
                    self._prev_humanoid_dof_vel = torch.zeros_like(self.env._humanoid_dof_vel)
                    self.jitter = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
                if not hasattr(self, "_prev_humanoid_dof_accel"):
                    self._prev_humanoid_dof_accel = torch.zeros_like(self.env._humanoid_dof_vel)
                    self.jerk = torch.zeros(self.env.num_envs, dtype=torch.float32, device=self.device)
                self._prev_humanoid_dof_vel[:] = self.env._humanoid_dof_vel
                self._prev_humanoid_dof_accel[:] = 0
                self.jitter[:] = 0
                self.jerk[:] = 0
            else:
                dof_accel = (self.env._humanoid_dof_vel - self._prev_humanoid_dof_vel) / self.env.dt
                mean_dof_accel = dof_accel.abs().mean(dim=-1)
                self.jitter[:] = self.jitter + mean_dof_accel
                if self._prev_humanoid_dof_accel.sum() == 0: # second frame
                    self._prev_humanoid_dof_accel[:] = dof_accel
                else:
                    dof_jerk = (dof_accel - self._prev_humanoid_dof_accel) / self.env.dt
                    mean_dof_jerk = dof_jerk.abs().mean(dim=-1)
                    self.jerk[:] = self.jerk + mean_dof_jerk

    def _env_reset_done(self):
        obs, done_env_ids = self.env.reset_done()
        if self.eval_jitter:
            self._collect_jitter_info(done_env_ids)
        return self.obs_to_torch(obs), done_env_ids

    def _post_step(self, info):
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            "actions_num": self.actions_num,
            "input_shape": obs_shape,
            "num_seqs": self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_value": self.normalize_value,
            "normalize_input": self.normalize_input,
        }
        return config

    def _setup_action_space(self):
        self.actions_num = self.action_space.shape[0]
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        return


    def maybe_capture_video(self):
        # (dglim - visualization)
        if self.env.capture_video:
            if self.env.episode_idx in self.env.target_episodes:
                self.env._encode_video(fps=60)
            self.env.frame_idx = 0
            self.env.episode_idx += 1
            if self.env.episode_idx > max(self.env.target_episodes):
                sys.exit()