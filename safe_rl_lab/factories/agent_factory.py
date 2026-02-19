from safe_rl_lab.agents.separate_actor_critic import SeparateActorCritic
from safe_rl_lab.agents.joint_actor_critic import JointActorCritic


class AgentFactory:

    @staticmethod
    def create(algo_type, obs_dim, act_dim, cfg, device):
        if algo_type == "ppo":
            pass

        elif algo_type == "ppo_lag":
            if cfg.algo.a2c_architecture == "separate":
                a2c = SeparateActorCritic(obs_dim, act_dim, cfg=cfg)
                return a2c
            elif cfg.algo.a2c_architecture == "shared":
                a2c = JointActorCritic(obs_dim, act_dim, cfg.algo.hidden_sizes, cfg.algo.activation,
                                       cfg.algo.output_activation, cfg.algo.weight_initialization_method, cfg)
                return a2c

        elif algo_type == "ppg":
            pass


        elif algo_type == "ppg_lag":
            if cfg.algo.a2c_architecture == "separate":
                a2c = SeparateActorCritic(obs_dim, act_dim, cfg=cfg)
                return a2c
            elif cfg.algo.a2c_architecture == "shared":
                pass

        else:
                raise ValueError(f"Unknown model type {cfg.algo.model_arch}")