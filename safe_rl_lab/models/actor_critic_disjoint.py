import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Normal

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCriticDisjoint(nn.Module):
    def __init__(self, envs, hidden_dim, squash_actions=True):
        super().__init__()
        self.squash_actions = squash_actions
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """
            Is logstd because of the exp -> can only be positive
        """
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            unsquashed = probs.rsample()
            squashed = torch.tanh(unsquashed)
        else:
            action = action.clamp(-0.999999, 0.999999)
            squashed = action
            unsquashed = torch.atanh(action)

        log_prob = probs.log_prob(unsquashed)
        log_prob = log_prob.sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(x)

        return squashed, log_prob, entropy, value