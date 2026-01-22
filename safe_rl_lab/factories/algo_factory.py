import torch
from safe_rl_lab.algo.ppo import PPO
from safe_rl_lab.algo.ppo_lag import PPOLag
from safe_rl_lab.factories.agent_factory import AgentFactory
from safe_rl_lab.utils.lagrange import PIDLagrange
from safe_rl_lab.utils.vector_runner import VectorRunner


class AlgoFactory:
    @staticmethod
    def create(cfg, env, logger, device):
        """
        Builds and returns the fully configured algorithm.
        """
        # ENV related
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        runner = VectorRunner(env, device, cfg)

        #Algorithm + Agent
        algo_type = cfg.algo.name.lower()

        agent = AgentFactory.create(algo_type, obs_dim, act_dim, cfg, device)

        if algo_type == "ppo":
            optimizer = torch.optim.Adam(agent.parameters(), lr=cfg.algo.lr, betas=(0.9, 0.999), eps=1e-5)

            return PPO(
                logger=logger,
                runner=runner,
                agent=agent,
                optimizer=optimizer,
                cfg=cfg,
                device=device,
            )
        elif algo_type == "ppo_lag":
            main_optimizer = torch.optim.Adam(agent.model.parameters(), lr=cfg.algo.lr)

            cost_lr = getattr(cfg.algo, "cost_lr", cfg.algo.lr)
            cost_optimizer = torch.optim.Adam(
                agent.cost_critic.parameters(),
                lr=cost_lr,
                eps=1e-5
            )

            return PPOLag(
                logger=logger,
                runner=runner,
                agent=agent,
                main_optimizer=main_optimizer,
                cost_optimizer=cost_optimizer,
                cfg=cfg,
                device=device,
            )
        else:
            raise NotImplementedError