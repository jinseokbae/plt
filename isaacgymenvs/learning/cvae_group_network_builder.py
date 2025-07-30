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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_games.algos_torch import layers, network_builder, torch_ext
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from isaacgymenvs.utils.quantizers import QuantizerSelector
from isaacgymenvs.learning.cvae_network_builder import SimpleEncoder, ContinuousEncoder, NullNet, ValueNet

class cVAEGroupBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            input_shape = kwargs.pop("input_shape")  # state / input for both enc and dec
            goal_input_shape = kwargs.pop("goal_input_shape")  # goal state / input for only enc
            latent_shape = kwargs.pop("latent_shape")  # latent / input for only dec
            actions_num = kwargs.pop("actions_num")
            self.actions_num = actions_num

            self.enc_type = kwargs.pop("enc_type")
            if self.enc_type in ["continuous", "hybrid"]:
                self.enc_scale = kwargs.pop("enc_scale")  # encoder fix scale
                self.continuous_enc_style = kwargs.pop("continuous_enc_style")  # continuous_enc_style
            if self.enc_type in ["discrete", "hybrid"]:
                self.code_num = kwargs.pop("code_num")
                self.quant_type = kwargs.pop("quant_type")
                self.num_quants = kwargs.pop("num_quants")
            if self.enc_type == "hybrid":
                self.post_cond_prior = kwargs.pop("post_cond_prior", False)

            self.value_size = kwargs.pop("value_size", 1)
            self.num_seqs = num_seqs = kwargs.pop("num_seqs", 1)

            # dof groups
            dof_group = kwargs.pop("dof_group")
            self.dof_group = dof_group
            # if two parts are exlcusively separated --> all 1, 
            # if some of the dof is shared among parts, than >1
            self.num_repetitions = torch.zeros(self.actions_num)
            for dof_ids in dof_group:
                self.num_repetitions[dof_ids] += 1
            actions_num_group = [len(group) for group in dof_group]
            self.num_groups = len(actions_num_group)
            if self.num_groups > 0:
                latent_dim_group = []
                _actions_num_remained = actions_num
                _latent_dim_remained = latent_shape[-1]
                for num in actions_num_group:
                    if sum(actions_num_group) > actions_num: # if some dofs are shared among parts
                        num = round(num * actions_num / sum(actions_num_group))
                    _latent_dim = round(_latent_dim_remained * (num / _actions_num_remained))
                    _actions_num_remained -= num
                    _latent_dim_remained -= _latent_dim
                    latent_dim_group.append(_latent_dim)
            else:
                latent_dim_group = [latent_shape[-1], ]
            self.latent_dim_group = latent_dim_group

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)

            self.load(params)
            self._build_actor_critic_net(
                input_shape=input_shape, goal_input_shape=goal_input_shape, latent_dim_group=latent_dim_group
            )
            actor_out_size = self.units["dec"][-1]
            critic_out_size = self.units["value"][-1]

            self.value = torch.nn.Linear(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                # self.logits = torch.nn.Linear(actor_out_size, actions_num)
                raise ValueError("cVAE policy must be continuous")
            if self.is_multi_discrete:
                # self.logits = torch.nn.ModuleList([torch.nn.Linear(actor_out_size, num) for num in actions_num])
                raise ValueError("cVAE policy must be continuous")
            if self.is_continuous:
                self.mu = torch.nn.ModuleList([torch.nn.Linear(actor_out_size, num) for num in actions_num_group])
                self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
                mu_init = self.init_factory.create(**self.space_config["mu_init"])
                self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])

                sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

                if not self.space_config["learn_sigma"]:
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False
                    )
                elif self.space_config["fixed_sigma"]:
                    self.sigma = nn.Parameter(
                        torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True
                    )
                else:
                    self.sigma = torch.nn.Linear(actor_out_size, actions_num)

            mlp_init = dict()
            for net in self.initializer.keys():
                mlp_init[net] = self.init_factory.create(**self.initializer[net])
            default_mlp_init = self.init_factory.create(**{"name": "default"})
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn["initializer"])

            for m_name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    net = m_name.split(".")[0]
                    if net == "encoder":
                        mlp_init["enc"](m.weight)
                    elif net == "decoder":
                        mlp_init["dec"](m.weight)
                    elif net == "critic_mlp":
                        mlp_init["value"](m.weight)
                    else:
                        default_mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            self.encoder.init_params()
            self.decoder.init_params()
            self.critic_mlp.init_params()

            if self.is_continuous:
                for mu in self.mu:
                    mu_init(mu.weight)
                if self.space_config["fixed_sigma"]:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            return

        def forward(self, obs_dict):
            obs = obs_dict["obs"]
            goal_obs = obs_dict["goal_obs"]
            actor_outputs = self.eval_actor(obs, goal_obs)
            value = self.eval_critic(obs, goal_obs)

            output = actor_outputs + (value, None)

            return output

        # get values from critic
        def eval_critic(self, obs, goal_obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out, goal_obs)
            value = self.value_act(self.value(c_out))
            return value

        # get parameters for action distribution
        def eval_actor(self, obs, goal_obs):
            if self.enc_type == "continuous":
                enc_dict = self.encoder(obs, goal_obs)
                info = dict(kl_loss=enc_dict["kl_loss"], prior_z=enc_dict['prior_z'], post_mu=enc_dict['post_mu'], prior_mu=enc_dict['prior_mu'])
                z = enc_dict["post_z"]
            elif self.enc_type == "deterministic":
                enc_dict = self.encoder(obs, goal_obs)
                info = dict(kl_loss=torch.zeros(1, device=obs.device), post_mu=enc_dict["z"])
                z = enc_dict["z"]
            elif self.enc_type == "discrete":
                enc_dict = self.encoder(obs, goal_obs)
                z_original = enc_dict["z"]
                z_original_splits = torch.split(z_original, self.latent_dim_group, dim=-1)
                z, indices, commit_loss = [], [], 0
                for i, split in enumerate(z_original_splits):
                    z_i, indices_i, commit_loss_i = self.quantizer[i](split[:, None])
                    z.append(z_i[:, 0])
                    indices.append(indices_i[:, 0])
                    commit_loss = commit_loss + commit_loss_i.mean(dim=-1)
                z = torch.cat(z, dim=-1)
                indices = torch.cat(indices, dim=-1)
                commit_loss = commit_loss / len(z_original_splits)
                info = dict(commit_loss=commit_loss, indices=indices, post_mu=z)
            elif self.enc_type == "hybrid":
                enc_dict = self.encoder(obs, goal_obs)
                info = dict(kl_loss=enc_dict["kl_loss"], commit_loss=enc_dict["commit_loss"], indices=enc_dict["indices"], prior_z=enc_dict['prior_z'], post_mu=enc_dict['post_mu'], prior_mu=enc_dict['prior_mu'])
                z = enc_dict["post_z"]

            info["latent"] = z 
            a_out_group = self.decoder(obs, z)

            # concat mu from each group
            mu = torch.zeros(obs.shape[0], self.actions_num, device=obs.device)
            for group_id, a_out in enumerate(a_out_group):
                _mu = self.mu_act(self.mu[group_id](a_out))
                target_dof_ids = self.dof_group[group_id]
                mu[:, target_dof_ids] = _mu
            mu = mu / self.num_repetitions[None].to(_mu.device)

            if self.space_config["fixed_sigma"]:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))

            return info, mu, sigma

        # only used for prior rollout
        def eval_actor_prior(self, obs, num_active_quants=1):
            if self.enc_type in ["continuous", "hybrid"]:
                if not self.post_cond_prior:
                    prior_out = self.encoder.encode_prior(obs)
                    prior_mu = prior_out["mu"]
                    prior_sigma = prior_out["sigma"]
                if self.enc_type == "hybrid":
                    if self.quant_type in ["rvq"]:
                        residual_mus = []
                        for idx in range(self.num_groups):
                            codebooks = self.encoder._post_quantizer[idx].codebooks[:num_active_quants]
                            uniform_indices = [torch.randint(low=0, high=self.code_num, size=(obs.shape[0], )).to(obs.device) for _ in range(num_active_quants)]
                            codes = [codebooks[k][ind] for k, ind in enumerate(uniform_indices)]
                            residual_mu = torch.stack(codes, dim=0).sum(dim=0)
                            residual_mus.append(residual_mu)
                        residual_mus = torch.cat(residual_mus, dim=-1)
                    else:
                        residual_mus = []
                        for idx in range(self.num_groups):
                            codebook = self.encoder._post_quantizer[idx].codebooks
                            uniform_indices = torch.randint(low=0, high=self.code_num, size=(obs.shape[0], )).to(obs.device)
                            residual_mu = codebook[uniform_indices] # non differentiable
                            residual_mus.append(residual_mu)
                        residual_mus = torch.cat(residual_mus, dim=-1)
                    if self.post_cond_prior:
                        prior_mu = self.encoder.encode_prior(obs, post_z=residual_mus)
                    prior_z = prior_mu + residual_mus
                else:
                    prior_dist = Normal(prior_mu, prior_sigma)
                    prior_z = prior_dist.rsample()

            elif self.enc_type == "discrete":
                if self.quant_type in ["rvq"]:
                    prior_z = []
                    for idx in range(self.num_groups):
                        codebooks = self.quantizer[idx].codebooks[:num_active_quants]
                        uniform_indices = [torch.randint(low=0, high=self.code_num, size=(obs.shape[0], )).to(obs.device) for _ in range(num_active_quants)]
                        codes = [codebooks[k][ind] for k, ind in enumerate(uniform_indices)]
                        _prior_z = torch.stack(codes, dim=0).sum(dim=0)
                        prior_z.append(_prior_z)
                    prior_z = torch.cat(prior_z, dim=-1)
                else:
                    prior_z = []
                    for idx in range(self.num_groups):
                        codebook = self.quantizer[idx].codebooks
                        uniform_indices = torch.randint(low=0, high=self.code_num, size=(obs.shape[0], )).to(obs.device)
                        _prior_z = codebook[uniform_indices] # non differentiable
                        prior_z.append(_prior_z)
                    prior_z = torch.cat(prior_z, dim=-1)
            a_outs = self.decoder(obs, prior_z)
            mu = []
            for idx, a_out in enumerate(a_outs):
                _mu = self.mu_act(self.mu[idx](a_out))
                mu.append(_mu)
            mu = torch.cat(mu, dim=-1)

            if self.space_config["fixed_sigma"]:
                sigma = mu * 0.0 + self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(a_out))
            return mu, sigma

        def load(self, params):
            ########################################
            self.separate = params.get("separate", False)

            # mlp arguments - encoder / decoder
            self.units = dict()
            self.activation = dict()
            self.initializer = dict()
            self.is_d2rl = dict()
            self.norm_only_first_layer = dict()
            for net in ["enc", "dec", "value"]:
                self.units[net] = params[net]["units"]
                self.activation[net] = params[net]["activation"]
                self.initializer[net] = params[net]["initializer"]
                self.is_d2rl[net] = params[net].get("d2rl", False)
                self.norm_only_first_layer[net] = params[net].get("norm_only_first_layer", False)

            # value net
            self.value_activation = params.get("value_activation", "None")
            self.normalization = params.get("normalization", None)
            self.has_rnn = "rnn" in params
            self.has_space = "space" in params
            self.central_value = params.get("central_value", False)
            self.joint_obs_actions_config = params.get("joint_obs_actions", None)

            if self.has_space:
                self.is_multi_discrete = "multi_discrete" in params["space"]
                self.is_discrete = "discrete" in params["space"]
                self.is_continuous = "continuous" in params["space"]
                if self.is_continuous:
                    self.space_config = params["space"]["continuous"]
                    self.fixed_sigma = self.space_config["fixed_sigma"]
                elif self.is_discrete:
                    self.space_config = params["space"]["discrete"]
                elif self.is_multi_discrete:
                    self.space_config = params["space"]["multi_discrete"]
            else:
                self.is_discrete = False
                self.is_continuous = False
                self.is_multi_discrete = False

            if self.has_rnn:
                self.rnn_units = params["rnn"]["units"]
                self.rnn_layers = params["rnn"]["layers"]
                self.rnn_name = params["rnn"]["name"]
                self.rnn_ln = params["rnn"].get("layer_norm", False)
                self.is_rnn_before_mlp = params["rnn"].get("before_mlp", False)
                self.rnn_concat_input = params["rnn"].get("concat_input", False)

            if "cnn" in params:
                self.has_cnn = True
                self.cnn = params["cnn"]
                self.permute_input = self.cnn.get("permute_input", True)
            else:
                self.has_cnn = False
            ########################################

            self._enc_units = params["enc"]["units"]
            self._enc_activation = params["enc"]["activation"]
            self._enc_initializer = params["enc"]["initializer"]
            return

        def _build_actor_critic_net(self, input_shape, goal_input_shape, latent_dim_group):
            # unused layers
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()

            # actor
            latent_dim = sum(latent_dim_group)
            if self.enc_type == "continuous":
                self.encoder = ContinuousEncoder(
                    state_dim=input_shape[-1],
                    goal_dim=goal_input_shape[-1],
                    latent_dim=latent_dim,
                    units=self.units["enc"],
                    activation=self.activations_factory.create(self.activation["enc"]),
                    initializer=self.init_factory.create(**self.initializer["enc"]),
                    fixed_scale=self.enc_scale,
                    style=self.continuous_enc_style,
                )
            elif self.enc_type == "hybrid":
                kwargs = dict(
                    code_num=self.code_num,
                    num_quants=self.num_quants,
                    quant_type=self.quant_type,
                    latent_dim_group=latent_dim_group,
                    post_cond_prior=self.post_cond_prior
                )
                assert self.continuous_enc_style in ["quantcond", "quantdirect", "quantdirectres"]
                self.encoder = HybridGroupEncoder(
                    state_dim=input_shape[-1],
                    goal_dim=goal_input_shape[-1],
                    latent_dim=latent_dim,
                    units=self.units["enc"],
                    activation=self.activations_factory.create(self.activation["enc"]),
                    initializer=self.init_factory.create(**self.initializer["enc"]),
                    style=self.continuous_enc_style,
                    kwargs=kwargs
                )
            elif self.enc_type == "deterministic":
                self.encoder = SimpleEncoder(
                    state_dim=input_shape[-1],
                    goal_dim=goal_input_shape[-1],
                    latent_dim=latent_dim,
                    units=self.units["enc"],
                    activation=self.activations_factory.create(self.activation["enc"]),
                    initializer=self.init_factory.create(**self.initializer["enc"]),
                )
            
            elif self.enc_type == "discrete":
                self.encoder = SimpleEncoder(
                    state_dim=input_shape[-1],
                    goal_dim=goal_input_shape[-1],
                    latent_dim=latent_dim,
                    units=self.units["enc"],
                    activation=self.activations_factory.create(self.activation["enc"]),
                    initializer=self.init_factory.create(**self.initializer["enc"]),
                )
                self.quantizer = nn.ModuleList([QuantizerSelector(
                    quant_type=self.quant_type,
                    code_num=self.code_num,
                    num_quants=self.num_quants,
                    dim=latent_dim_i 
                ) for latent_dim_i in latent_dim_group])

            self.decoder = GroupMLPDecoder(
                state_dim=input_shape[-1],
                latent_dim_group=latent_dim_group,
                units=self.units["dec"],
                activation=self.activations_factory.create(self.activation["dec"]),
                initializer=self.init_factory.create(**self.initializer["dec"]),
            )

            # critic
            self.critic_mlp = ValueNet(
                state_dim=input_shape[-1],
                goal_dim=goal_input_shape[-1],
                units=self.units["value"],
                activation=self.activations_factory.create(self.activation["value"]),
                initializer=self.init_factory.create(**self.initializer["value"]),
            )
            return

    def build(self, name, **kwargs):
        net = cVAEGroupBuilder.Network(self.params, **kwargs)
        return net


class HybridGroupEncoder(nn.Module):
    def __init__(
        self, state_dim, goal_dim, latent_dim, units, activation, initializer, style, kwargs=None
    ):
        super().__init__()
        self._units = units
        self._initializer = initializer
        self.latent_dim_group = kwargs.pop("latent_dim_group")
        self._style = style

        # post net
        layers = []
        in_size = state_dim + goal_dim
        for out_size in units:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation)
            in_size = out_size
        self._post_net = nn.Sequential(*layers)
        self._post_loc_net = nn.Linear(units[-1], latent_dim)

        # prior net
        self._post_cond_prior = kwargs.get("post_cond_prior", False)
        if self._post_cond_prior:
            in_size = state_dim + latent_dim
        else:
            in_size = state_dim
        layers = []
        for out_size in units:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation)
            in_size = out_size
        self._prior_net = nn.Sequential(*layers)
        self._prior_loc_net = nn.Linear(units[-1], latent_dim)
        self._vae_logstd_max = 1 # for stability
        self._post_quantizer = nn.ModuleList([QuantizerSelector(
            quant_type = kwargs['quant_type'],
            code_num = kwargs['code_num'],
            num_quants = kwargs['num_quants'],
            dim = latent_dim,
        ) for latent_dim in self.latent_dim_group])

        # init
        self.init_params()

    def forward(self, obs, goal):
        prior_input = obs
        post_input = torch.cat([obs, goal], dim=-1)

        # extract posterior mu and sigma
        post_feat = self._post_net(post_input)
        post_loc = self._post_loc_net(post_feat)

        """Forward Posterior and Prior Network"""
        if self._style == "quantcond":
            # extract prior mu and sigma
            prior_feat = self._prior_net(prior_input)
            prior_loc = self._prior_loc_net(prior_feat)

            # quantize residual 
            quant_target = post_loc[:, None] - prior_loc[:, None].detach()
        elif self._style in ["quantdirect", "quantdirectres"]:
            quant_target = post_loc[:, None]

        quant_splits = torch.split(quant_target, self.latent_dim_group, dim=-1)
        quants, indices, commit_loss = [], [], 0
        for i, split in enumerate(quant_splits):
            quant_i, indices_i, commit_loss_i = self._post_quantizer[i](split)
            quant_i = quant_i[:, 0]
            indices_i = indices_i[:, 0]
            commit_loss_i = commit_loss_i.mean(dim=-1)
            # append
            quants.append(quant_i)
            indices.append(indices_i)
            commit_loss = commit_loss + commit_loss_i

        quants = torch.cat(quants, dim=-1)
        indices = torch.cat(indices, dim=-1)
        commit_loss = commit_loss / len(quant_splits)
        if self._style in ["quantdirect", "quantdirectres"]:
            if self._post_cond_prior:
                prior_input = torch.cat([prior_input, quants], dim=-1)
            # extract prior mu and sigma
            prior_feat = self._prior_net(prior_input)
            prior_loc = self._prior_loc_net(prior_feat)

        """Calculate loss"""
        if self._style == "quantcond":
            # override
            post_loc = prior_loc.detach() + quants # straight-through : maintaining post loc's gradient
            kl_loss = (post_loc - prior_loc).pow(2).sum(dim=-1)
        elif self._style in ["quantdirect", "quantdirectres"]:
            # override
            post_loc = prior_loc + quants
            if self._style == "quantdirect":
                kl_loss = prior_loc.pow(2).sum(dim=-1)
            elif self._style == "quantdirectres":
                quant_error = (quant_target[:, 0] - quants)
                kl_loss = (prior_loc - quant_error.detach()).pow(2).sum(dim=-1)

        # margin minimizing loss
        post_z = post_loc
        prior_z = prior_loc

        out_dict = dict(post_z=post_z, prior_z=prior_z, kl_loss=kl_loss, post_mu=post_loc, prior_mu=prior_loc, indices=indices, commit_loss=commit_loss)
        return out_dict
    
    # forward prior parameters
    def encode_prior(self, obs, post_z=None):
        if post_z is not None:
            assert self._post_cond_prior
            prior_input = torch.cat([obs, post_z], dim=-1)
        else:
            prior_input = obs
        prior_feat = self._prior_net(prior_input)
        prior_loc = self._prior_loc_net(prior_feat)
        prior_output = dict(
            mu=prior_loc,
            sigma=torch.zeros_like(prior_loc)
        )
        return prior_output

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return

class GroupMLPDecoder(nn.Module):
    def __init__(self, state_dim, latent_dim_group, units, activation, initializer, residual=True):
        super().__init__()
        self._residual = residual
        self._units = units
        self._initializer = initializer
        self._latent_dim_group = latent_dim_group

        self._net = []
        if residual:
            self._activation = []

        for latent_dim in latent_dim_group:
            layers = []
            if residual:
                in_size = state_dim
                for out_size in units:
                    layers.append(nn.Linear(in_size + latent_dim, out_size))
                    in_size = out_size
                _net = nn.ModuleList(layers)
                _activation = activation
                self._net.append(_net)
                self._activation.append(_activation)
            else:
                in_size = state_dim + latent_dim
                for out_size in units:
                    layers.append(nn.Linear(in_size, out_size))
                    layers.append(activation)
                    in_size = out_size
                _net = nn.Sequential(*layers)
                self._net.append(_net)
        
        self._net = nn.ModuleList(self._net)
        if residual:
            self._activation = nn.ModuleList(self._activation)
        # init
        self.init_params()

    def forward(self, obs, latent): # (conditioning latent for every layer)
        outs = []
        latent_splits = torch.split(latent, self._latent_dim_group, dim=-1) # split into groups
        for group_id, latent_split in enumerate(latent_splits):
            if self._residual:
                input = obs
                for i in range(len(self._units)):
                    aug_input = torch.cat([input, latent_split], dim=-1)
                    out = self._activation[group_id](self._net[group_id][i](aug_input))
                    input = out
                outs.append(out)
            else:
                input = torch.cat([obs, latent_split], dim=-1)
                outs.append(self._net[group_id](input))
        return outs

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return
