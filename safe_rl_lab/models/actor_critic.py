import torch
import torch.nn as nn
from torch.distributions import Normal
from abc import ABC, abstractmethod
from safe_rl_lab.utils.model_utils import build_mlp_network

class ActorCriticBase(nn.Module, ABC):
    """
    Doesnt care about PPO or PPG. Just: Input -> Dist, Value
    """

    @abstractmethod
    def forward(self, obs):
        """:returns (distribution, value prediction)"""
        pass


class SharedActorCritic(ActorCriticBase):
    """Shared Backbone Architecture
    :arg obs_dim: int
    :arg act_dim: int
    :arg hidden_sizes: List of hidden sizes
    :arg activation: str
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.backbone = build_mlp_network([obs_dim] + hidden_sizes, hidden_sizes, activation)

        self.actor_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        hidden = self.backbone(obs)

        val = self.critic_head(hidden)

        mu = self.actor_head(hidden)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        return dist, val


class DisjointActorCritic(ActorCriticBase):
    """Disjoint Architecture (Preferred for PPG)"""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.actor_net = build_mlp_network(sizes=[obs_dim] + hidden_sizes + [act_dim], activation=activation)
        self.critic_net = build_mlp_network(sizes=[obs_dim] + hidden_sizes + [1], activation=activation)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        # Value
        val = self.critic_net(obs)

        # Policy
        mu = self.actor_net(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        return dist, val