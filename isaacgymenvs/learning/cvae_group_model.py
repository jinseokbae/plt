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
from learning.cvae_model import ModelcVAEContinuous


class ModelcVAEGroupContinuous(ModelcVAEContinuous):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        return super().build(config)

    class Network(ModelcVAEContinuous.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self.latent_dim_group = a2c_network.latent_dim_group
            return

        # Only for downstream task
        # input by current prioprioceptive observation (obs) and latent from encoder or high-level policy (latent)
        def decode(self, obs, latent):
            _, *X = latent.shape

            # TODO: implement case when latent is indices

            # forward latent and observation to decoder
            z = latent
            obs = self.norm_obs(obs)
            a_out_group = self.a2c_network.decoder(obs, z)

            mu = torch.zeros(obs.shape[0], self.a2c_network.actions_num, device=obs.device)
            for idx, a_out in enumerate(a_out_group):
                _mu = self.a2c_network.mu_act(self.a2c_network.mu[idx](a_out))
                target_dof_ids = self.a2c_network.dof_group[idx]
                mu[:, target_dof_ids] = _mu
            mu = mu / self.a2c_network.num_repetitions[None].to(_mu.device)

            if self.a2c_network.space_config["fixed_sigma"]:
                logstd = mu * 0.0 + self.a2c_network.sigma_act(self.a2c_network.sigma)
            else:
                logstd = self.a2c_network.sigma_act(self.a2c_network.sigma(a_out))

            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            action = distr.sample()
            return action, mu
