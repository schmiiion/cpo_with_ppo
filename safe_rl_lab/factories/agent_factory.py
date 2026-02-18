from safe_rl_lab.agents.constraint_actor_critic import SeparateActorCritic
from safe_rl_lab.models.actor_critic import GaussianActor
from safe_rl_lab.models.critics import Critic


class AgentFactory:

    @staticmethod
    def create(algo_type, obs_dim, act_dim, cfg, device):
        if algo_type == "ppo":
            if cfg.algo.a2c_architecture == "separate":
                actor = GaussianActor(obs_dim, act_dim, cfg.algo.hidden_sizes,
                                      cfg.algo.activation, cfg.algo.output_activation, cfg.algo.weight_initialization_method)
                value_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)
                a2c = SeparateActorCritic(actor=actor, reward_critic=value_critic, cfg=cfg)
                return a2c
            else:
                pass


        elif algo_type == "ppo_lag":
            if cfg.algo.a2c_architecture == "separate":
                actor = GaussianActor(obs_dim, act_dim, cfg.algo.hidden_sizes, cfg.algo.activation,
                                      cfg.algo.output_activation,
                                      cfg.algo.weight_initialization_method)
                reward_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)
                cost_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)
                a2c = SeparateActorCritic(actor=actor, reward_critic=reward_critic, cost_critic=cost_critic, cfg=cfg)
                return a2c
            else:
                pass

        elif algo_type == "ppg":
            pass


        elif algo_type == "ppg_lag":
            pass

        else:
                raise ValueError(f"Unknown model type {cfg.algo.model_arch}")