from abc import ABC, abstractmethod
import torch

class PolicyGradient(ABC):
    """
    agent is a container to hold the policy and value function. Either shared or disjoint.
    """
    def __init__(self, agent, logger, cfg, device="cpu"):
        self.agent = agent
        self.device = device
        self.logger = logger
        self.cfg = cfg

    @abstractmethod
    def update(self, buffer):
        """
        Takes a buffer of experience and updates the model.
        Must be implemented by PPO, PPG, TRPO, etc.
        """
        pass

    def get_action(self, obs):
        """Interface for the Runner to interact with the Agent."""
        with torch.no_grad():
            if isinstance(obs, torch.Tensor):
                obs = obs.to(self.device)
            else:
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

            return self.agent.get_action(obs)