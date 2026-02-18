import torch
import torch.nn as nn
from safe_rl_lab.models.actor_critic import GaussianActor
from safe_rl_lab.utils.model_utils import build_mlp_network
from torch.distributions import Normal


class JointActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation, weight_initialization_mode):
        super().__init__()
        self.encoder = build_mlp_network(
            sizes=[obs_dim] + hidden_sizes,
            activation=activation,
            weight_initialization_mode=weight_initialization_mode)

        if output_activation == "tanh":
            self.actor_head = nn.Sequential([nn.Linear(hidden_sizes[-1], act_dim), nn.Tanh()])
        else:
            self.actor_head = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5, requires_grad=True)
        self.reward_critic_head = nn.Linear(hidden_sizes[-1], 1)
        self.cost_critic_head = nn.Linear(hidden_sizes[-1], 1)

        #init optimizer
        lr = self.cfg.algo.lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)


################################################################################
################################## INTERFACES ##################################
################################################################################

    def step(self, obs):
        """used by the runner to get the next action the respective logp and the value estimate."""
        with torch.no_grad():
            hidden = self.encoder(obs)

            mu = self.actor_head(hidden)
            std = torch.exp(self.log_std)
            pd = Normal(mu, std)

            val = self.reward_critic_head(hidden)

            action = pd.sample()
            logp = pd.log_prob(action).sum(-1)


            agent_info = {"val": val, "logp": logp}

        return action, agent_info

    def evaluate_actions(self, obs, act):
        """returns new logp, entropy, value and cost value predictions"""
        hidden = self.encoder(obs)

        mu = self.actor_head(hidden)
        std = torch.exp(self.log_std)
        pd = Normal(mu, std)

        action = pd.sample()
        logp = pd.log_prob(action).sum(dim=-1)
        entropy = pd.entropy().sum(dim=-1)

        reward_prediction = self.reward_critic_head(hidden)
        cost_prediction = self.cost_critic_head(hidden)
        return logp, entropy, reward_prediction, cost_prediction

    # --- PPG Specific Methods ---

    def get_aux_value(self, obs):
        """For PPG Aux Phase"""
        return self.actor.forward_aux(obs)

    def get_aux_cost(self, obs):
        """For PPG Lag Aux Phase"""
        if hasattr(self.actor, "forward_aux_cost"):
            return self.actor.forward_aux_cost(obs)
        return None