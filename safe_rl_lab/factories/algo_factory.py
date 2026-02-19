from safe_rl_lab.algo.ppg_lag import PPGLag
from safe_rl_lab.algo.ppo import PPO
from safe_rl_lab.algo.ppo_lag import PPOLag
from safe_rl_lab.factories.agent_factory import AgentFactory
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
        algo_type: str = cfg.algo.name.lower()
        a2c_container = AgentFactory.create(algo_type, obs_dim, act_dim, cfg, device)

        if algo_type == "ppo":
            return PPO(
                logger=logger,
                runner=runner,
                a2c=a2c_container,
                cfg=cfg,
                device=device,
            )

        elif algo_type == "ppo_lag":

            return PPOLag(
                logger=logger,
                runner=runner,
                agent=a2c_container,
                cfg=cfg,
                device=device,
            )

        elif algo_type == "ppg_lag":
            return PPGLag(
                logger=logger,
                runner=runner,
                a2c=a2c_container,
                cfg=cfg,
                device=device,
            )

        else:
            raise NotImplementedError