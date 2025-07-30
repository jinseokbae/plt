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
from rl_games.algos_torch import layers, network_builder, torch_ext
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.common import object_factory
from rl_games.algos_torch import torch_ext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
from rl_games.algos_torch.d2rl import D2RLNet
from rl_games.algos_torch.sac_helper import SquashedNormal
from rl_games.common.layers.recurrent import GRUWithDones, LSTMWithDones


class HierMultiDiscreteBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)
            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            self.actor_cnn = nn.Sequential()
            self.critic_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()
            self.critic_mlp = nn.Sequential()

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)
            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            mlp_args = {
                'input_size' : in_mlp_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            # TODO
            # self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            self.logits = HierarchicalLogits(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    
            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)                    

                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)
                            
                value = self.value_act(self.value(c_out))

                # TODO
                logits = self.logits(a_out)
                return logits, value, states

            else:
                out = obs
                out = self.actor_cnn(out)
                out = out.flatten(1)                

                out = self.actor_mlp(out)
                value = self.value_act(self.value(out))

                # TODO
                logits = self.logits(out)
                return logits, value, states

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)
            value = self.value_act(self.value(c_out))
            return value

    def build(self, name, **kwargs):
        net = HierMultiDiscreteBuilder.Network(self.params, **kwargs)
        return net


class HierarchicalLogits(nn.Module):
    def __init__(self, in_size, actions_num, use_rnn=True, hidden_size=512, proj_size=512):
        super().__init__()
        self.use_rnn = use_rnn
        self.n_layers = len(actions_num)

        if use_rnn:
            proj_size = hidden_size
            self._projs = nn.ModuleList([nn.Linear(in_size, proj_size) for _ in actions_num])
            self.rnn_cell = nn.GRUCell(input_size=proj_size, hidden_size=hidden_size)
            self.init_state = nn.Parameter(torch.randn(1, hidden_size))
            self._logits = nn.ModuleList([nn.Linear(hidden_size + proj_size, num) for num in actions_num])
        else:
            _layers = []
            for idx, num in enumerate(actions_num):
                if idx == 0:
                    _layers.append(nn.Linear(in_size, num))
                else:
                    prev_num = actions_num[idx - 1]
                    _layers.append(nn.Linear(in_size + prev_num, num))
            self._logits = torch.nn.ModuleList(_layers)
    
    def forward(self, input):
        outs = []
        if self.use_rnn:
            B, _ = input.shape
            prev_state = self.init_state.expand(B, -1)
            for i in range(self.n_layers):
                # projection
                proj = self._projs[i](input)
                # logit output
                logit_input = torch.cat([proj, prev_state], dim=-1)
                logit = self._logits[i](logit_input)
                outs.append(logit)
                # rnn state update
                rnn_input = proj
                prev_state = self.rnn_cell(rnn_input, prev_state)
        else:
            for idx, _logit in enumerate(self._logits):
                if idx == 0:
                    outs.append(_logit(input))
                else:
                    aug_input = torch.cat([input, outs[-1]], dim=-1)
                    outs.append(_logit(aug_input))
        return outs