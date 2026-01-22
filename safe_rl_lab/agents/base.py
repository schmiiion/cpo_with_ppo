import torch.nn as nn
import torch

class Agent(nn.Module):
    """Base interface for all agents."""

    def act(self, obs, return_dist_params=False): #wanted during sampling
        """Plain PPO interaction"""
        with torch.no_grad():
            pd, val = self.model(obs)
            val = val.flatten()
            action = pd.sample()
            logp = pd.log_prob(action).sum(-1)

        info = {"val": val, "logp": logp} #value is of shape [num_envs, 1]

        if return_dist_params:
            info["pd_mean"] = pd.loc
            info["pd_std"] = pd.scale

        return action, info

    def get_value(self, obs):
        _, val = self.model(obs)
        return val

    def evaluate_actions(self, obs, actions):
        pd, val = self.model(obs)

        log_probs = pd.log_prob(actions).sum(-1)

        entropy = pd.entropy().sum(-1)

        return log_probs, entropy, val.flatten()
