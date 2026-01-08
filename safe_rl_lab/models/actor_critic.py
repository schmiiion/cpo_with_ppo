import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np
from safe_rl_lab.models.abstractBaseClasses import PpoModel

#####Erkenntnisse:
# Trennung Optimization und Inferenz
# Rollout: Collect Data of policy, NOT training -> kein Gradient Tracking

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(PpoModel):

    def __init__(self, *, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
          layer_init(nn.Linear(obs_dim, hidden_dim)),
          nn.Tanh(),
          layer_init(nn.Linear(hidden_dim, hidden_dim)),
          nn.Tanh(),
        )
        self.actor_mean = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.actor_log_std = nn.Parameter(torch.zeros(1, act_dim))
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1)


    def forward(self, obs):
        h = self.backbone(obs)
        action_mean = self.actor_mean(h)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_log_std = action_log_std.clamp(-20, 2)
        action_std = torch.exp(action_log_std)
        pdf = Normal(action_mean, action_std)

        vpred = self.critic(h)
        return pdf, vpred, None