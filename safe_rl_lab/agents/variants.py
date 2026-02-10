from wandb.integration.torch.wandb_torch import torch
import torch
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
    def __init__(self, model, value_critic):
        super().__init__(model)
        self.value_critic = value_critic

    def act(self, obs):
        """
        Overrides the base class's act()
        - Calls model, i.e. Actor for the action/ logp
        - Calls disjoint Value Critic for the value estimate
        """
        with torch.no_grad():
            pd, _ = self.model(obs)
            action = pd.sample()
            logp = pd.log_prob(action).sum(-1)

            val = self.value_critic(obs).flatten()

        info = {"val": val, "logp": logp}
        return action, info

    def get_value(self, obs):
        """Override base get_value to use the correct critic"""
        return self.value_critic(obs)

    def evaluate_actions(self, obs, actions, need_val=True):
        pd, _ = self.model(obs)

        log_probs = pd.log_prob(actions).sum(-1)

        entropy = pd.entropy().sum(-1)

        if need_val:
            val = self.value_critic(obs).flatten()
        else:
            val = None

        return log_probs, entropy, val

    def get_policy_value(self, obs):
        """
        Specific to PPG: Get values from the policy's aux head vs value net.
        """
        policy_val = self.model.forward_aux(obs)
        return policy_val

class PPGLagAgent(PPGAgent):
    """
    Requires a DisjointActorCritic to perform aux updates
    """
    def __init__(self, model, value_critic, cost_critic):
        super().__init__(model, value_critic)
        self.cost_critic = cost_critic

    def get_cost_value(self, obs):
        return self.cost_critic(obs).flatten()

    def get_policy_cost(self, obs):
        policy_cost = self.model.forward_aux_cost(obs)
        return policy_cost