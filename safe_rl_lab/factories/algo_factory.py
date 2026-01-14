import torch
from safe_rl_lab.algo.ppo_lag import PPOLag
from safe_rl_lab.algo.ppo import PPO
from safe_rl_lab.models.actor_critic import ActorCritic


class AlgoFactory:
    @staticmethod
    def create(cfg, envs, device):
        """
        Builds and returns the fully
        """