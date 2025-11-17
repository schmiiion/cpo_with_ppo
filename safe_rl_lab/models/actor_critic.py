import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np


#####Erkenntnisse:
# Trennung Optimization und Inferenz
# Rollout: Collect Data of policy, NOT training -> kein Gradient Tracking
# Env schneidet actions bei -1, 1 ab, also selbst und nachvollziehbar machen!
#

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
          nn.Linear(obs_dim, hidden_dim),
          nn.Tanh(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, act_dim * 2)
        self.critic = nn.Linear(hidden_dim, 1)


    def forward_actor(self, obs):
        #Idea: std independent of state/ parameter tying
        out = self.actor(self.backbone(obs))
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2) # keep it sane
        std = torch.exp(log_std)
        return mu, std

    def forward_critic(self, obs):
        out = self.critic(self.backbone(obs))
        return out.squeeze(-1)

    #rollout-time: sample squashed action + corrected logp
    @torch.no_grad()
    def sample_action_and_logp(self, obs):
        mu, std = self.forward_actor(obs)
        base = Normal(mu, std)
        z = base.rsample() #same as in VAE
        a = torch.tanh(z)
        #log prob with tanh correction
        logp = base.log_prob(z) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(-1)
        return a, logp

    #PPO update time: logp of action under new policy
    def log_prob_of_action(self, obs, action):
        a = action.clamp(-1 + 1e-6, 1 - 1e-6)
        pre_tanh = 0.5 * ((1+a).log() - (1-a).log())
        mu, std = self.forward_actor(obs)
        base = Normal(mu, std)
        logp = base.log_prob(pre_tanh) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(-1)
        return logp
