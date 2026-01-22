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
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode):
        super().__init__()
        self.backbone = build_mlp_network(sizes=[obs_dim] + hidden_sizes, activation=activation, weight_initialization_mode=weight_initialization_mode)

        self.actor_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, obs):
        hidden = self.backbone(obs)

        val = self.critic_head(hidden)

        mu = self.actor_head(hidden)
        log_std_clamped = torch.clamp(self.log_std, min=-2.0, max=2.0)
        std = torch.exp(log_std_clamped)
        dist = Normal(mu, std)

        return dist, val


class DisjointActorCritic(ActorCriticBase):
    """Disjoint Architecture"""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode):
        super().__init__()
        self.actor_net = build_mlp_network(sizes=[obs_dim] + hidden_sizes + [act_dim], activation=activation, weight_initialization_mode=weight_initialization_mode)
        self.critic_net = build_mlp_network(sizes=[obs_dim] + hidden_sizes + [1], activation=activation, weight_initialization_mode=weight_initialization_mode)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, obs):
        # Value
        val = self.critic_net(obs)

        # Policy
        mu = self.actor_net(obs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        return dist, val

class PPGActorCritic(ActorCriticBase):
    """Disjoint Actor and Critic. The Actor has a policy and a value head."""

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, weight_initialization_mode):
        super().__init__()
        self.encoder = build_mlp_network(sizes=[obs_dim] + hidden_sizes, activation=activation,
                                           weight_initialization_mode=weight_initialization_mode)
        self.actor_head = build_mlp_network(sizes=[hidden_sizes[-1]] + [act_dim], activation=activation,
                                            weight_initialization_mode=weight_initialization_mode)
        self.aux_head = build_mlp_network(sizes=[hidden_sizes[-1], 1], activation=activation,)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5)

    def forward(self, obs):
        h = self.encoder(obs)
        mu = self.actor_head(h)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)

        return dist, None

    def forward_aux(self, obs):
        h = self.encoder(obs)
        val = self.aux_head(h)
        return val