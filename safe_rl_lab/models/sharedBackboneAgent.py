import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Normal
from safe_rl_lab.models.abstractBaseClasses import PpoModel

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def _get_env_dims(env):
    # Handle both single and vectorized envs
    if hasattr(env, "single_observation_space"):
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        obs_space = env.observation_space
        act_space = env.action_space

    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(np.prod(act_space.shape))
    return obs_dim, act_dim

class SharedBackboneAgent(PpoModel):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        obs_dim, act_dim = _get_env_dims(envs)

        self.backbone = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        #policy head -> actor
        self.actor_mean = _layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        # self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -1.0))

        #value head -> critic
        self.critic = _layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _features(self, x):
        if x.dim() > 2:
            print('input flattening of the observations triggered - did the observation space change?')
            x = x.view(x.size(0), -1)
        return self.backbone(x)

    def forward(self, obs) -> "pd, vpred, aux":
        h = self._features(obs)

        action_mean = self.actor_mean(h)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = action_logstd.exp()
        dist = Normal(action_mean, action_std)

        vpred = self.critic(h)

        return dist, vpred, None


    # def get_value(self, x):
    #     h = self._features(x)
    #     return self.critic(h)
    #
    # def get_action_and_value(self, x, action=None):
    #     h = self._features(x)
    #
    #     action_mean = self.actor_mean(h)
    #     action_logstd = self.actor_logstd.expand_as(action_mean)
    #     action_std = action_logstd.exp()
    #     dist = Normal(action_mean, action_std)
    #
    #     if action is None:
    #         action = dist.sample()
    #
    #     log_prob = dist.log_prob(action).sum(1)
    #     entropy = dist.entropy().sum(1)
    #     value = self.critic(h)
    #     return action, log_prob, entropy, value

