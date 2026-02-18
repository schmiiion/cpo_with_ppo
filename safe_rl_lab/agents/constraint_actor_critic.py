import torch
import torch.nn as nn
from safe_rl_lab.models.actor_critic import GaussianActor


class SeparateActorCritic(nn.Module):
    def __init__(
            self,
            actor: nn.Module,
            reward_critic: nn.Module = None,
            cost_critic: nn.Module = None,
            cfg = None
    ):
        super().__init__()
        self.cfg = cfg

        #Model assignment
        self.actor = actor
        self.reward_critic = reward_critic
        self.cost_critic = cost_critic

        # #shared architecture quirk
        # if isinstance(self.actor, SharedActorCritic) and self.reward_critic is None:
        #     self.reward_critic = self.actor

        #init optimizers
        self.actor_optimizer = None
        self.reward_critic_optimizer = None
        self.cost_critic_optimizer = None

        self._init_optimizers()

    def _init_optimizers(self):
        """Intelligently build optimizers based on architecture type."""
        lr = self.cfg.algo.lr

        # # --- A. Shared Backbone Architecture ---> Actor and Critic use the same backbone
        # if isinstance(self.actor, SharedActorCritic):
        #     """shared"""
        #
        # # --- B. PPG / Safe PPG Architecture ---
        # elif isinstance(self.actor, (PPGActorCritic, SafePPGActorCritic)):
        #     """PPG"""
        #     pass
        # # --- C. Disjoint / Standard Architecture ---
        # else:

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        if self.reward_critic is not None and self.reward_critic != self.actor:
            self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=lr)

        # --- D. Cost Critic (Always Disjoint/Separate) ---
        if self.cost_critic is not None:
            cost_lr = getattr(self.cfg.algo, "cost_lr", lr)
            self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=cost_lr)

################################################################################
################################## INTERFACES ##################################
################################################################################

    def step(self, obs):
        """used by the runner to get the next action the respective logp and the value estimate."""
        with torch.no_grad():
            if isinstance(self.actor, list):
                """PPG"""
            elif isinstance(self.actor, list):
                """Shared architecture"""
            else:
                # Separate PPO
                pd = self.actor(obs)
                val = self.reward_critic(obs).flatten()

            action = pd.sample()
            logp = pd.log_prob(action).sum(-1)
            agent_info = {"val": val, "logp": logp}

        return action, agent_info


    def get_value(self, obs):
        """used during the optimization of the value critic"""
        if isinstance(self.actor, list):
            _, val = self.actor(obs)
            return val
        return self.reward_critic(obs)

    def get_cost_value(self, obs):
        """used during the optimization of the cost critic"""
        if self.cost_critic is None:
            raise ValueError("No cost critic defined!")
        return self.cost_critic(obs)

    def evaluate_actions(self, obs, act):
        """evaluate actions based on current policy"""
        if isinstance(self.actor, (GaussianActor)):
            dist = self.actor(obs)
        else:
            raise ValueError("Non-Gaussian actor used")

        # 2. Calculate statistics for the EXISTING actions
        # sum(-1) sums over the action dimension (e.g. for continuous control)
        logp = dist.log_prob(act).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return logp, entropy

    # --- PPG Specific Methods ---

    def get_aux_value(self, obs):
        """For PPG Aux Phase"""
        return self.actor.forward_aux(obs)

    def get_aux_cost(self, obs):
        """For PPG Lag Aux Phase"""
        if hasattr(self.actor, "forward_aux_cost"):
            return self.actor.forward_aux_cost(obs)
        return None