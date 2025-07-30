import torch
import torch.nn as nn
from torch import einsum
from torch.distributions import Categorical
import torch.nn.functional as F
from rl_games.algos_torch.models import ModelA2C, ModelA2CContinuousLogStd, ModelA2CMultiDiscrete
from rl_games.algos_torch.torch_ext import CategoricalMasked
from vector_quantize_pytorch import FSQ
from vector_quantize_pytorch.vector_quantize_pytorch import gumbel_sample

class ModelCommonContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build("common", **config)
        for name, _ in net.named_parameters():
            print(name)

        obs_shape = config["input_shape"]
        normalize_value = config.get("normalize_value", False)
        normalize_input = config.get("normalize_input", False)
        value_size = config.get("value_size", 1)

        return self.Network(
            net,
            obs_shape=obs_shape,
            normalize_value=normalize_value,
            normalize_input=normalize_input,
            value_size=value_size,
        )

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
                return result
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }
                return result

class ModelCommonDiscrete(ModelA2CMultiDiscrete):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):
        net = self.network_builder.build("common", **config)
        for name, _ in net.named_parameters():
            print(name)

        obs_shape = config["input_shape"]
        normalize_value = config.get("normalize_value", False)
        normalize_input = config.get("normalize_input", False)
        value_size = config.get("value_size", 1)

        if net.is_multi_discrete:
            return self.MultiDiscreteNetwork(
                net,
                obs_shape=obs_shape,
                normalize_value=normalize_value,
                normalize_input=normalize_input,
                value_size=value_size,
            )
        else:
            return self.SingleDiscreteNetwork(
                net,
                obs_shape=obs_shape,
                normalize_value=normalize_value,
                normalize_input=normalize_input,
                value_size=value_size,
            )

    class SingleDiscreteNetwork(ModelA2C.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)

            if is_train:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                prev_neglogp = -categorical.log_prob(prev_actions)
                entropy = categorical.entropy()
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : categorical.logits,
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states
                }
                gumbel = input_dict.get('gumbel', False)
                if gumbel:
                    gumbel_temperature = input_dict.get('gumbel_temperature', 1.0)
                    gumbel_stochastic = input_dict.get('gumbel_stochastic', False)
                    _, one_hots = self._gumbel_sample(logits, gumbel_temperature, gumbel_stochastic)
                    result['one_hots'] = one_hots
            else:
                categorical = CategoricalMasked(logits=logits, masks=action_masks)
                selected_action = categorical.sample().long()
                neglogp = -categorical.log_prob(selected_action)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'logits' : categorical.logits,
                    'rnn_states' : states
                }
            return  result

        def _gumbel_sample(self, logits, temperature, stochastic):
            ind, one_hot = gumbel_sample(
                logits=logits,
                temperature=temperature,
                stochastic=stochastic,
                straight_through=True,
                reinmax=True,
                dim=-1,
                training=True)
            return ind, one_hot

    class MultiDiscreteNetwork(ModelA2CMultiDiscrete.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            return

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            action_masks = input_dict.get('action_masks', None)
            prev_actions = input_dict.get('prev_actions', None)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            logits, value, states = self.a2c_network(input_dict)
            if action_masks is None:
                categorical = [Categorical(logits=logit) for logit in logits]
            else:   
                categorical = [CategoricalMasked(logits=logit, masks=mask) for logit, mask in zip(logits, action_masks)]

            # whether training or not
            if is_train:
                prev_actions = torch.split(prev_actions, 1, dim=-1)
                prev_neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, prev_actions)]
                prev_neglogp = torch.stack(prev_neglogp, dim=-1).sum(dim=-1)
                entropy = [c.entropy() for c in categorical]
                entropy = torch.stack(entropy, dim=-1).sum(dim=-1)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'logits' : [c.logits for c in categorical],
                    'values' : value,
                    'entropy' : torch.squeeze(entropy),
                    'rnn_states' : states
                }
                gumbel = input_dict.get('gumbel', False)
                if gumbel:
                    gumbel_temperature = input_dict.get('gumbel_temperature', 1.0)
                    gumbel_stochastic = input_dict.get('gumbel_stochastic', False)
                    _, one_hots = self._gumbel_sample(logits, gumbel_temperature, gumbel_stochastic)
                    result['one_hots'] = one_hots
            else:
                selected_action = [c.sample().long() for c in categorical]
                neglogp = [-c.log_prob(a.squeeze()) for c,a in zip(categorical, selected_action)]
                selected_action = torch.stack(selected_action, dim=-1)
                neglogp = torch.stack(neglogp, dim=-1).sum(dim=-1)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.unnorm_value(value),
                    'actions' : selected_action,
                    'logits' : [c.logits for c in categorical],
                    'rnn_states' : states
                }
            return  result

        def _gumbel_sample(self, logits, temperature, stochastic):
            inds, one_hots = [], []
            for logit in logits:
                ind, one_hot = gumbel_sample(
                    logits=logit,
                    temperature=temperature,
                    stochastic=stochastic,
                    straight_through=True,
                    reinmax=True,
                    dim=-1,
                    training=True)
                inds.append(ind)
                one_hots.append(one_hot)

            one_hots = torch.stack(one_hots, dim=1)
            return inds, one_hots