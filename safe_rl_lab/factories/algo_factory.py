import torch
from safe_rl_lab.algo.ppo_lag import PPOLag
from safe_rl_lab.algo.ppo import PPO
from safe_rl_lab.models.actor_critic import ActorCritic
from safe_rl_lab.models.cost_critic import CostCritic
from safe_rl_lab.utils.lagrange import Lagrange


class AlgoFactory:
    @staticmethod
    def create(cfg, envs, device):
        """
        Builds and returns the fully configured algorithm.
        """
        obs_dim = envs.single_observation_space.shape[0]
        action_dim = envs.single_action_space.shape[0]

        if cfg.model.arch == "shared":
            agent = ActorCritic().to(device)
        else:
            raise NotImplementedError(f"{cfg.model.arch} not implemented")

        optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.algo.lr)

        algo_type = cfg.algo.name.lower()

        if algo_type == "ppo":
            return PPO(
                agent=agent,
                optimizer=optimizer,
                device=device,
                cfg=cfg
            )
        elif algo_type == "ppo_lag":
            cost_critic = CostCritic()
            cost_optimizer = torch.optim.Adam(cost_critic.parameters(), lr=cfg.algo.lr)

            lagrange = Lagrange()


            return PPOLag()

        else:
            raise ValueError(f"{algo_type} not known")