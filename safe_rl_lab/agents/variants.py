from safe_rl_lab.agents.base import Agent

class PPOAgent(Agent):
    """
    Vanilla PPO Agent. Accepts EITHER SharedActorCritic OR DisjointActorCritic
    """

    def __init__(self, model):
        super().__init__()
        self.model = model


class SafePPOAgent(PPOAgent):
    """
    PPO + Safety.
    Composition: PPOModel + Cost Critic
    """

    def __init__(self, model, cost_critic):
        super().__init__(model)
        self.cost_critic = cost_critic

    def get_cost_value(self, obs):
        return self.cost_critic(obs).flatten()


class PPGAgent(PPOAgent):
    """
    Requires a DisjointActorCritic to perform aux updates
    """
    def __init__(self, model):
        super().__init__(model)

    def get_aux_prediction(self, obs):
        """
        Specific to PPG: Get values from the policy's aux head vs value net.
        """
        # ... PPG specific logic ...
        pass