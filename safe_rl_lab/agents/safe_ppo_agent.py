from safe_rl_lab.agents.ppo_agent import PPOAgent


class SafePPOAgent(PPOAgent):
    """
    PPO + Safety.
    Composition: PPOModel + Cost Critic
    """
    def __init__(self, model, cost_critic):
        super().__init__(model)
        self.cost_critic = cost_critic

    def get_cost_value(self, obs):
        return self.cost_critic(obs)