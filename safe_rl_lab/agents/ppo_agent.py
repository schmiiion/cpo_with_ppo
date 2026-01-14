from safe_rl_lab.agents.agent import Agent


class PPOAgent(Agent):
    """
    Vanilla PPO Agent. Accepts EITHER SharedActorCritic OR DisjointActorCritic
    """

    def __init__(self, model):
        super().__init__()
        self.model = model