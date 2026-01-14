import torch.nn as nn
import torch

class Agent(nn.Module):
    """Base interface for all agents."""

    def act(self, obs, return_dist_params=False):
        """Plain PPO interaction"""
        with torch.no_grad():
            pd, val = self.model(obs)
            action = pd.sample()
            logp = pd.log_prob(action).sum(-1)

        info = {"val": val, "logp": logp}

        if return_dist_params:
            info["pd_mean"] = pd.loc
            info["pd_std"] = pd.scale

        return action, info

    def get_value(self, obs):
        _, val = self.model(obs)
        return val