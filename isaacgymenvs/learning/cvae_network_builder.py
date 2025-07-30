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

EVALUATE_BINS = False
class cVAEBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            input_shape = kwargs.pop("input_shape")  # state / input for both enc and dec
            goal_input_shape = kwargs.pop("goal_input_shape")  # goal state / input for only enc
            latent_shape = kwargs.pop("latent_shape")  # latent / input for only dec
            actions_num = kwargs.pop("actions_num")
            self._no_goal = kwargs.pop("no_goal")

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

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)

            self.load(params)
            self._build_actor_critic_net(
                input_shape=input_shape, goal_input_shape=goal_input_shape, latent_shape=latent_shape
            )
            actor_out_size = self.units["dec"][-1]
            critic_out_size = self.units["value"][-1]

            self.value = torch.nn.Linear(critic_out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(actor_out_size, actions_num)
            """
                for multidiscrete actions num is a tuple
            """
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(actor_out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(actor_out_size, actions_num)
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
                mu_init(self.mu.weight)
                if self.space_config["fixed_sigma"]:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            if EVALUATE_BINS: # fort latent visualization
                self._bins = torch.zeros(self.quantizer.codebook_size)
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
                z, indices, commit_loss = self.quantizer(z_original[:, None])
                z = z[:, 0] # squeeze seq space
                indices = indices[:, 0]
                info = dict(commit_loss=commit_loss, indices=indices, post_mu=z)
                if EVALUATE_BINS:
                    for indice in indices:
                        self._bins[indice.item()] += 1
            elif self.enc_type == "hybrid":
                enc_dict = self.encoder(obs, goal_obs)
                info = dict(kl_loss=enc_dict["kl_loss"], commit_loss=enc_dict["commit_loss"], indices=enc_dict["indices"], prior_z=enc_dict['prior_z'], post_mu=enc_dict['post_mu'], prior_mu=enc_dict['prior_mu'])
                z = enc_dict["post_z"]

            info["latent"] = z 
            a_out = self.decoder(obs, z)

            if self.is_discrete:
                logits = self.logits(a_out)
                return info, logits

            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return info, logits

            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.space_config["fixed_sigma"]:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))

                return info, mu, sigma
            return

        # only used for prior rollout
        def eval_actor_prior(self, obs, num_active_quants=1):
            if self.enc_type in ["continuous", "hybrid"]:
                if not self.post_cond_prior:
                    prior_out = self.encoder.encode_prior(obs)
                    prior_mu = prior_out["mu"]
                    prior_sigma = prior_out["sigma"]
                if self.enc_type == "hybrid":
                    if self.quant_type in ["rvq"]:
                        codebooks = self.encoder._post_quantizer.codebooks[:num_active_quants]
                        uniform_indices = [torch.randint(low=0, high=self.code_num, size=(obs.shape[0], )).to(obs.device) for _ in range(num_active_quants)]
                        codes = [codebooks[k][ind] for k, ind in enumerate(uniform_indices)]
                        residual_mu = torch.stack(codes, dim=0).sum(dim=0)
                    else:
                        codebook = self.encoder._post_quantizer.codebooks
                        uniform_indices = torch.randint(low=0, high=self.code_num, size=(obs.shape[0], )).to(obs.device)
                        residual_mu = codebook[uniform_indices] # non differentiable
                    if self.post_cond_prior:
                        prior_mu = self.encoder.encode_prior(obs, post_z=residual_mu)
                    prior_z = prior_mu + residual_mu
                else:
                    prior_dist = Normal(prior_mu, prior_sigma)
                    prior_z = prior_dist.rsample()
            elif self.enc_type == "discrete":
                if self.quant_type in ["rvq"]:
                    codebooks = self.quantizer.codebooks[:num_active_quants]
                    uniform_indices = [torch.randint(low=0, high=self.code_num, size=(obs.shape[0], )).to(obs.device) for _ in range(num_active_quants)]
                    codes = [codebooks[k][ind] for k, ind in enumerate(uniform_indices)]
                    prior_z = torch.stack(codes, dim=0).sum(dim=0)
                else:
                    codebook = self.quantizer.codebooks
                    uniform_indices = torch.randint(low=0, high=self.code_num, size=(obs.shape[0], )).to(obs.device)
                    prior_z = codebook[uniform_indices] # non differentiable
            a_out = self.decoder(obs, prior_z)
            mu = self.mu_act(self.mu(a_out))
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

        def _build_actor_critic_net(self, input_shape, goal_input_shape, latent_shape):
            # unused layers
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()

            # actor
            if self.enc_type == "continuous":
                self.encoder = ContinuousEncoder(
                    state_dim=input_shape[-1],
                    goal_dim=goal_input_shape[-1],
                    latent_dim=latent_shape[-1],
                    units=self.units["enc"],
                    activation=self.activations_factory.create(self.activation["enc"]),
                    initializer=self.init_factory.create(**self.initializer["enc"]),
                    fixed_scale=self.enc_scale,
                    style=self.continuous_enc_style,
                    kwargs=dict(),
                )
            elif self.enc_type == "hybrid":
                kwargs = dict(
                    code_num=self.code_num,
                    num_quants=self.num_quants,
                    quant_type=self.quant_type,
                    post_cond_prior=self.post_cond_prior
                )
                assert self.continuous_enc_style in ["quantcond", "quantdirect", "quantdirectres"]
                self.encoder = ContinuousEncoder(
                    state_dim=input_shape[-1],
                    goal_dim=goal_input_shape[-1],
                    latent_dim=latent_shape[-1],
                    units=self.units["enc"],
                    activation=self.activations_factory.create(self.activation["enc"]),
                    initializer=self.init_factory.create(**self.initializer["enc"]),
                    fixed_scale=self.enc_scale,
                    style=self.continuous_enc_style,
                    kwargs=kwargs
                )
            elif self.enc_type == "deterministic":
                if self._no_goal:
                    self.encoder = NullNet()
                else:
                    self.encoder = SimpleEncoder(
                        state_dim=input_shape[-1],
                        goal_dim=goal_input_shape[-1],
                        latent_dim=latent_shape[-1],
                        units=self.units["enc"],
                        activation=self.activations_factory.create(self.activation["enc"]),
                        initializer=self.init_factory.create(**self.initializer["enc"]),
                    )
            
            elif self.enc_type == "discrete":
                self.encoder = SimpleEncoder(
                    state_dim=input_shape[-1],
                    goal_dim=goal_input_shape[-1],
                    latent_dim=latent_shape[-1],
                    units=self.units["enc"],
                    activation=self.activations_factory.create(self.activation["enc"]),
                    initializer=self.init_factory.create(**self.initializer["enc"]),
                )
                self.quantizer = QuantizerSelector(
                    quant_type=self.quant_type,
                    code_num=self.code_num,
                    num_quants=self.num_quants,
                    dim=latent_shape[-1]
                )

            if self._no_goal:
                self.decoder = MLP(
                    state_dim=input_shape[-1],
                    units=self.units["dec"],
                    activation=self.activations_factory.create(self.activation["dec"]),
                    initializer=self.init_factory.create(**self.initializer["dec"]),
                )
            else:
                self.decoder = MLPDecoder(
                    state_dim=input_shape[-1],
                    latent_dim=latent_shape[-1],
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
        net = cVAEBuilder.Network(self.params, **kwargs)
        return net


class ContinuousEncoder(nn.Module):
    def __init__(
        self, state_dim, goal_dim, latent_dim, units, activation, initializer, style="controlvae", fixed_scale=0.5, kwargs=None
    ):
        super().__init__()
        self._units = units
        self._initializer = initializer
        self._style = style  # ["controlvae"(Heyuan Yao et al. 2022), "standard"(Won et al. 2022), "statecond"]
        if self._style == "controlvae":
            self._scale = fixed_scale
        elif self._style in ["standard", "statecond", "quantcond", "quantdirect", "quantdirectres"]:
            self._scale = 1.0  # not used
        else:
            raise NotImplementedError("invalid style")

        # prior net
        if self._style == "controlvae":
            # post net
            layers = []
            in_size = state_dim + goal_dim
            for out_size in units:
                layers.append(nn.Linear(in_size, out_size))
                layers.append(activation)
                in_size = out_size
            layers.append(nn.Linear(units[-1], latent_dim))
            self._post_net = nn.Sequential(*layers)

            layers = []
            in_size = state_dim
            for out_size in units:
                layers.append(nn.Linear(in_size, out_size))
                layers.append(activation)
                in_size = out_size
            layers.append(nn.Linear(units[-1], latent_dim))
            self._prior_net = nn.Sequential(*layers)
        elif self._style == "standard":
            # post net
            layers = []
            in_size = state_dim + goal_dim
            for out_size in units:
                layers.append(nn.Linear(in_size, out_size))
                layers.append(activation)
                in_size = out_size
            layers.append(nn.Linear(units[-1], latent_dim * 2))
            self._post_net = nn.Sequential(*layers)
        elif self._style in ["statecond", 'quantcond', 'quantdirect', 'quantdirectres']:
            # post net
            layers = []
            in_size = state_dim + goal_dim
            for out_size in units:
                layers.append(nn.Linear(in_size, out_size))
                layers.append(activation)
                in_size = out_size
            self._post_net = nn.Sequential(*layers)
            self._post_loc_net = nn.Linear(units[-1], latent_dim)
            if self._style == "statecond":
                self._post_scale_net = nn.Linear(units[-1], latent_dim)
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
            if self._style == "statecond":
                self._prior_scale_net = nn.Linear(units[-1], latent_dim)
            self._vae_logstd_max = 1 # for stability
            if self._style in ['quantcond', 'quantdirect', 'quantdirectres']:
                self._post_quantizer = QuantizerSelector(
                    quant_type = kwargs['quant_type'],
                    code_num = kwargs['code_num'],
                    num_quants = kwargs['num_quants'],
                    dim = latent_dim,
                )
        # init
        self.init_params()
        if self._style == "statecond":
            with torch.no_grad():
                self._post_scale_net.weight.uniform_(-1.0, 0.0)
                self._prior_scale_net.weight.uniform_(-1.0, 0.0)

    def forward(self, obs, goal):
        prior_input = obs
        post_input = torch.cat([obs, goal], dim=-1)
        if self._style == "controlvae":
            prior_loc = self._prior_net(prior_input)
            prior_scale = torch.ones_like(prior_loc) * self._scale
            prior_dist = Normal(prior_loc, prior_scale)

            post_loc = self._post_net(post_input)
            post_loc = prior_loc + post_loc
            post_scale = prior_scale.clone()
            post_dist = Normal(post_loc, post_scale)

            post_z = post_dist.rsample()
            prior_z = prior_dist.rsample()

            kl_loss = self.controlvae_kl_loss(post_loc=post_loc, prior_loc=prior_loc).mean(dim=-1)
        elif self._style == "standard":
            post_loc, post_scale = torch.chunk(self._post_net(post_input), 2, dim=-1)
            post_scale = F.softplus(post_scale) + 1e-6
            post_dist = Normal(post_loc, post_scale)

            prior_loc = torch.zeros_like(post_loc)
            prior_scale = torch.ones_like(post_scale)
            prior_dist = Normal(prior_loc, prior_scale)

            post_z = post_dist.rsample()
            kl_loss = kl_divergence(post_dist, prior_dist).mean(dim=-1)

        elif self._style in ["statecond", "quantcond", "quantdirect", "quantdirectres"]:
            # extract posterior mu and sigma
            post_feat = self._post_net(post_input)
            post_loc = self._post_loc_net(post_feat)

            """Forward Posterior and Prior Network"""
            if self._style == "statecond":
                # extract prior mu and sigma
                prior_feat = self._prior_net(prior_input)
                prior_loc = self._prior_loc_net(prior_feat)

            # quantize residual 
            elif self._style == "quantcond":
                # extract prior mu and sigma
                prior_feat = self._prior_net(prior_input)
                prior_loc = self._prior_loc_net(prior_feat)

                post_prior_loc_gap = post_loc[:, None] - prior_loc[:, None].detach()
                post_prior_loc_gap, indices, commit_loss = self._post_quantizer(post_prior_loc_gap)
                post_prior_loc_gap = post_prior_loc_gap[:, 0]
                indices = indices[:, 0]
                # override
                post_loc = prior_loc.detach() + post_prior_loc_gap # straight-through : maintaining post loc's gradient
            elif self._style in ["quantdirect", "quantdirectres"]:
                quant_target = post_loc[:, None]
                post_loc, indices, commit_loss = self._post_quantizer(quant_target)
                post_loc = post_loc[:, 0] # quantized
                indices = indices[:, 0]
                quant_error = quant_target[:, 0] - post_loc

                if self._post_cond_prior:
                    prior_input = torch.cat([prior_input, post_loc], dim=-1)

                # extract prior mu and sigma
                prior_feat = self._prior_net(prior_input)
                prior_loc = self._prior_loc_net(prior_feat)

                post_loc = post_loc + prior_loc

            """Calculate loss"""
            if self._style == "statecond":
                # scales
                post_logstd = self._post_scale_net(post_feat)
                post_logstd = torch.clamp(post_logstd, -5, self._vae_logstd_max) # following PHC
                post_scale = torch.exp(post_logstd) + 1e-6
                prior_logstd = self._prior_scale_net(prior_feat)
                prior_logstd = torch.clamp(prior_logstd, -5, self._vae_logstd_max) # following PHC
                prior_scale = torch.exp(prior_logstd) + 1e-6

                # make dist
                post_dist = Normal(post_loc, post_scale)
                prior_dist = Normal(prior_loc, prior_scale)

                # sample
                post_z = post_dist.rsample()
                prior_z = prior_dist.rsample()
                kl_loss = kl_divergence(post_dist, prior_dist).mean(dim=-1)

            elif self._style == "quantcond":
                post_z = post_loc
                prior_z = prior_loc
                kl_loss = (post_loc - prior_loc).pow(2).sum(dim=-1)

            elif self._style in ["quantdirect", "quantdirectres"]:
                post_z = post_loc
                prior_z = prior_loc
                if self._style == "quantdirect":
                    kl_loss = prior_loc.pow(2).sum(dim=-1)
                elif self._style == "quantdirectres":
                    kl_loss = (prior_loc - quant_error.detach()).pow(2).sum(dim=-1)

        out_dict = dict(post_z=post_z, prior_z=prior_z, kl_loss=kl_loss, post_mu=post_loc, prior_mu=prior_loc)
        if self._style in ["quantcond", "quantdirect", "quantdirectres"]:
            out_dict.update(dict(
                indices=indices,
                commit_loss=commit_loss
            ))
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
        if self._style == "statecond":
            prior_logstd = self._prior_scale_net(prior_feat)
            prior_logstd = torch.clamp(prior_logstd, -5, self._vae_logstd_max) # following PHC
            prior_scale = torch.exp(prior_logstd) + 1e-6
            prior_output = dict(
                mu=prior_loc,
                sigma=prior_scale,
            )
        else:
            prior_output = dict(
                mu=prior_loc,
                sigma=torch.zeros_like(prior_loc)
            )
        return prior_output

    # only using for controlvae setting
    def controlvae_kl_loss(self, post_loc, prior_loc):
        return 0.5 * (post_loc - prior_loc) ** 2 / (self._scale**2)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return

class SimpleEncoder(nn.Module): # (no VAE type)
    # Referenced by https://github.com/Mael-zys/T2M-GPT/blob/main/models/vqvae.py
    def __init__(self, state_dim, goal_dim, latent_dim, units, activation, initializer):
        super().__init__()
        self._units = units
        self._initializer = initializer

        # enc net
        layers = []
        in_size = state_dim + goal_dim
        for out_size in units:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation)
            in_size = out_size
        layers.append(nn.Linear(units[-1], latent_dim))
        self._net = nn.Sequential(*layers)

        # init
        self.init_params()

    def forward(self, obs, goal):
        enc_input = torch.cat([obs, goal], dim=-1)
        z = self._net(enc_input)

        out_dict = dict(z=z)
        return out_dict

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return

class NullNet(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, obs, latent):
        return dict(z=None)
    
    def init_params(self):
        return

class MLP(nn.Module):
    def __init__(self, state_dim, units, activation, initializer):
        super().__init__()
        self._units = units
        self._initializer = initializer
        layers = []

        in_size = state_dim
        for out_size in units:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation)
            in_size = out_size
        self._net = nn.Sequential(*layers)
        # init
        self.init_params()

    def forward(self, obs, latent):
        input = obs
        return self._net(input)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return

class MLPDecoder(nn.Module):
    def __init__(self, state_dim, latent_dim, units, activation, initializer, residual=True):
        super().__init__()
        self._residual = residual
        self._units = units
        self._initializer = initializer
        layers = []

        if residual:
            in_size = state_dim
            for out_size in units:
                layers.append(nn.Linear(in_size + latent_dim, out_size))
                in_size = out_size
            self._net = nn.ModuleList(layers)
            self._activation = activation
        else:
            in_size = state_dim + latent_dim
            for out_size in units:
                layers.append(nn.Linear(in_size, out_size))
                layers.append(activation)
                in_size = out_size
            self._net = nn.Sequential(*layers)
        # init
        self.init_params()

    def forward(self, obs, latent): # (conditioning latent for every layer)
        if self._residual:
            input = obs
            for i in range(len(self._units)):
                aug_input = torch.cat([input, latent], dim=-1)
                out = self._activation(self._net[i](aug_input))
                input = out
            return out
        else:
            input = torch.cat([obs, latent], dim=-1)
            return self._net(input)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return


class ValueNet(nn.Module):
    def __init__(self, state_dim, goal_dim, units, activation, initializer):
        super().__init__()
        self._units = units
        self._initializer = initializer

        layers = []
        in_size = state_dim + goal_dim
        for out_size in units:
            layers.append(nn.Linear(in_size, out_size))
            layers.append(activation)
            in_size = out_size
        self._net = nn.Sequential(*layers)

        # init
        self.init_params()

    def forward(self, obs, goal_obs):
        input = torch.cat([obs, goal_obs], dim=-1)
        return self._net(input)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                self._initializer(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
        return






