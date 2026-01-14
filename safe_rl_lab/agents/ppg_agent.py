from safe_rl_lab.agents.ppo_agent import PPOAgent
from safe_rl_lab.models.actor_critic import DisjointActorCritic


class PPGAgent(PPOAgent):
    """
    Requires a DisjointActorCritic to perform aux updates
    """
    def __init__(self, model):
        super().__init__(model)
        assert isinstance(model, DisjointActorCritic),"PPG requires disjoint architecture!"

    def get_aux_prediction(self, obs):
        """
        Specific to PPG: Get values from the policy's aux head vs value net.
        """
        # ... PPG specific logic ...
        pass