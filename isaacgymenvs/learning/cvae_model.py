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
import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd


class ModelcVAEContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build("cvae", **config)
        for name, _ in net.named_parameters():
            print(name)

        obs_shape = config["input_shape"]
        normalize_value = config.get("normalize_value", False)
        normalize_input = config.get("normalize_input", False)
        value_size = config.get("value_size", 1)

        #
        enc_type = config.get("enc_type", "continuous")

        return self.Network(
            net,
            obs_shape=obs_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size,
            enc_type=enc_type,
        )

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            self.enc_type = kwargs.pop("enc_type")
            super().__init__(a2c_network, **kwargs)
            return

        # Model forward function
        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            info, mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
                if "kl_loss" in info:
                    result["vae_kl_loss"] = info["kl_loss"].mean()
                if "commit_loss" in info:
                    result["vae_commit_loss"] = info["commit_loss"].mean()
                if "post_mu" in info:
                    result["post_mu"] = info["post_mu"]
                if "prior_mu" in info:
                    result["prior_mu"] = info["prior_mu"]
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    "neglogpacs": torch.squeeze(neglogp),
                    "values": self.unnorm_value(value),
                    "actions": selected_action,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                    "indices": info.get("indices", None),
                    "latents": info['latent'] # latent input by decoder (low level controller)
                }
                if "post_mu" in info:
                    result["post_mu"] = info["post_mu"]
                if "prior_mu" in info:
                    result["prior_mu"] = info["prior_mu"]
            return result

        # (Aborted) Forward only latent - Latent Supervision for Continuous High-level policy
        def forward_latent(self, input_dict):
            obs = self.norm_obs(input_dict["obs"])
            goal_obs = input_dict["goal_obs"]
            info, *_ = self.a2c_network.eval_actor(obs, goal_obs) 
            latent = info["latent"]
            result = {
                "latents": latent,
            }
            return result

        # (Aborted) Forward only latent - Latent Supervision for Discrete High-level policy
        def forward_indices(self, input_dict):
            obs = self.norm_obs(input_dict["obs"])
            goal_obs = input_dict["goal_obs"]
            info, *_ = self.a2c_network.eval_actor(obs, goal_obs) 
            indices = info["indices"]
            result = {
                "indices": indices,
            }
            return result
        
        # Forward prior mu and sigma from the continuous (standalone) prior network
        # cVAE and ours only
        def forward_prior(self, input_dict):
            obs = self.norm_obs(input_dict["obs"])
            if hasattr(self.a2c_network ,'post_cond_prior'):
                if self.a2c_network.post_cond_prior:
                    return self.a2c_network.encoder.encode_prior(obs, input_dict["post_z"])
                else:
                    return self.a2c_network.encoder.encode_prior(obs)
            else:
                return self.a2c_network.encoder.encode_prior(obs)

        # Used only for test
        # conduct prior sampling (prior latents) for all the models
        @torch.no_grad()
        def prior_rollout(self, input_dict):
            # First output prior inference
            obs = self.norm_obs(input_dict["obs"])
            num_active_quants = input_dict.get("num_active_quants", 1)
            mu, sigma = self.a2c_network.eval_actor_prior(obs, num_active_quants)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            selected_action = distr.sample()
            result = dict(
                mus=mu,
                actions=selected_action,
                rnn_states=None
            )
            return result


        # Only for downstream task
        # input by current prioprioceptive observation (obs) and latent from encoder or high-level policy (latent)
        def decode(self, obs, latent):
            _, *X = latent.shape

            if self.enc_type == "discrete": # for VQ model, indices -> retreived code
                if latent.dtype == torch.int64:
                    if self.a2c_network.quant_type == "fsq":
                        z = self.a2c_network.quantizer.indices_to_codes(latent)
                    else:
                        if len(latent.shape) == 1 and not self.a2c_network.quant_type == "simple":
                            latent = latent[:, None]
                        z = self.a2c_network.quantizer.get_codes_from_indices(latent)
                        if self.a2c_network.quant_type in ["rvq", "rvq_cosine", "rqvq"]:
                            z = z.sum(dim=0)
                elif latent.dtype == torch.float32:
                    z, _, _ = self.a2c_network.quantizer(latent)
                else:
                    raise ValueError("Invalid dtype for latent")
            # for cVAE and ours
            # note that our high-level policy output should be concatednated with prior mu in advance
            # therefore, even if our high-level policy is discrete policy, the incoming latent is continuous variable (retreived)
            else:
                z = latent

            # forward latent and observation to decoder
            obs = self.norm_obs(obs)
            a_out = self.a2c_network.decoder(obs, z)

            mu = self.a2c_network.mu_act(self.a2c_network.mu(a_out))
            if self.a2c_network.space_config["fixed_sigma"]:
                logstd = mu * 0.0 + self.a2c_network.sigma_act(self.a2c_network.sigma)
            else:
                logstd = self.a2c_network.sigma_act(self.a2c_network.sigma(a_out))

            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            action = distr.sample()
            return action, mu
