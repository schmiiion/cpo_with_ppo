from safe_rl_lab.agents.variants import PPOAgent, SafePPOAgent, PPGAgent, PPGLagAgent
from safe_rl_lab.models.actor_critic import SharedActorCritic, DisjointActorCritic, PPGActorCritic, SafePPGActorCritic
from safe_rl_lab.models.critics import Critic


class AgentFactory:

    @staticmethod
    def create(algo_type, obs_dim, act_dim, cfg, device):
        if algo_type == "ppg":
            model = PPGActorCritic(obs_dim, act_dim, cfg.algo.hidden_sizes, cfg.algo.activation,
                                   cfg.algo.initialization_method)
        elif algo_type == "ppg_lag":
            model = SafePPGActorCritic(obs_dim, act_dim, cfg.algo.hidden_sizes, cfg.algo.activation,
                                       cfg.algo.initialization_method)
        elif cfg.algo.model_arch == "shared":
            model = SharedActorCritic(obs_dim, act_dim, cfg.algo.hidden_sizes, cfg.algo.activation, cfg.algo.initialization_method)
        elif cfg.algo.model_arch == "disjoint":
            model = DisjointActorCritic(obs_dim, act_dim, cfg.algo.hidden_sizes, cfg.algo.activation, cfg.algo.initialization_method)
        else:
            raise ValueError(f"Unknown model type {cfg.algo.model_arch}")

        model.to(device)

        if algo_type == "ppo":
            agent = PPOAgent(model)
        elif algo_type == "ppo_lag":
            cost_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)
            cost_critic.to(device)
            agent = SafePPOAgent(model, cost_critic)
        elif algo_type == "ppg":
            assert isinstance(model, PPGActorCritic), "PPG requires a specific ActorCritic!"
            value_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)
            value_critic.to(device)
            agent = PPGAgent(model, value_critic)
        elif algo_type == "ppg_lag":
            value_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)
            value_critic.to(device)
            cost_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)
            cost_critic.to(device)
            agent = PPGLagAgent(model, value_critic, cost_critic)


        else:
            raise ValueError(f"Unknown algo_type: {algo_type}")


        return agent