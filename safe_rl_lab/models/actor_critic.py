import torch.nn as nn
import torch


class ActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super().__init__()
        self.backbone = nn.Sequential(
          nn.Linear(obs_dim, hidden_dim),
          nn.GELU(),
          nn.Linear(hidden_dim, hidden_dim),
          nn.GELU(),
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
        return out
