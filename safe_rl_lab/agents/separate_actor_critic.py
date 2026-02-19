import torch
import torch.nn as nn
from safe_rl_lab.utils.model_utils import build_mlp_network
from torch.distributions import Normal


class SeparateActorCritic(nn.Module):
    def __init__(
            self,
            obs_dim,
            act_dim,
            cfg = None
    ):
        super().__init__()
        self.cfg = cfg
        self.is_ppg = "ppg" in cfg.algo.name.lower()

        if self.is_ppg:
            self.actor = GaussianActor(obs_dim, act_dim, cfg.algo.hidden_sizes, cfg.algo.activation,
                                       cfg.algo.output_activation, cfg.algo.weight_initialization_method, separate_heads=True)
        else:
            self.actor = GaussianActor(obs_dim, act_dim, cfg.algo.hidden_sizes, cfg.algo.activation,
                                       cfg.algo.output_activation, cfg.algo.weight_initialization_method)

        self.reward_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)
        self.cost_critic = Critic(obs_dim, cfg.algo.hidden_sizes, cfg.algo.activation)

        #init optimizers
        lr = cfg.algo.lr
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.reward_critic_optimizer = torch.optim.Adam(self.reward_critic.parameters(), lr=lr)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=lr)


################################################################################
################################## INTERFACES ##################################
################################################################################

    def step(self, obs):
        """used by the runner to get the next action the respective logp and the value estimate."""
        with torch.no_grad():
            pd = self.actor(obs)
            val = self.reward_critic(obs).flatten()

            action = pd.sample()
            logp = pd.log_prob(action).sum(-1)
            agent_info = {"val": val, "logp": logp}

        return action, agent_info


    def get_value(self, obs):
        """used during the optimization of the value critic"""
        return self.reward_critic(obs)

    def get_cost_value(self, obs):
        """used during the optimization of the cost critic"""
        return self.cost_critic(obs)

    def get_distribution(self, obs):
        return self.actor(obs)

    def evaluate_actions(self, obs, act):
        """evaluate actions based on current policy"""
        dist = self.actor(obs)
        logp = dist.log_prob(act).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return logp, entropy

    # --- PPG Specific Methods ---
    def get_aux_reward(self, obs):
        return self.actor.get_aux_reward(obs)

    def get_aux_cost(self, obs):
        return self.actor.get_aux_cost(obs)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation, weight_initialization_mode, separate_heads=False):
        super().__init__()
        self.separate_heads = separate_heads
        if not separate_heads:
            self.actor_net = build_mlp_network(sizes=[obs_dim] + hidden_sizes + [act_dim],
                                               activation=activation,
                                               output_activation=output_activation,
                                               weight_initialization_mode=weight_initialization_mode)
        else:
            self.actor_encoder = build_mlp_network(sizes=[obs_dim] + hidden_sizes,
                                                   activation=activation,
                                                   weight_initialization_mode=weight_initialization_mode)
            self.policy_head = nn.Linear(hidden_sizes[-1], act_dim)
            self.reward_head = nn.Linear(hidden_sizes[-1], 1)
            self.cost_head = nn.Linear(hidden_sizes[-1], 1)

        self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5, requires_grad=True)

    def forward(self, obs):
        if self.separate_heads:
            hidden = self.actor_encoder(obs)
            mu = self.policy_head(hidden)
        else:
            mu = self.actor_net(obs)

        log_std = torch.clamp(self.log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        return dist

    def get_aux_reward(self, obs):
        hidden = self.actor_encoder(obs)
        return self.reward_head(hidden).flatten()

    def get_aux_cost(self, obs):
        hidden = self.actor_encoder(obs)
        return self.cost_head(hidden).flatten()


class Critic(nn.Module):
    def __init__(
            self,
            obs_dim,
            hidden_sizes,
            activation='tanh',
            weight_initialization_mode='kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`CostCritic`."""
        nn.Module.__init__(self)
        self._obs_dim = obs_dim
        self._weight_initialization_mode = weight_initialization_mode
        self._activation = activation
        self._hidden_sizes = hidden_sizes

        self.net = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, 1],
            activation=self._activation,
            weight_initialization_mode=self._weight_initialization_mode,
        )

    def forward(self, obs: torch.Tensor):
        """Forward function.

        Specifically, V function approximator maps observations to V-values.

        Args:
            obs (torch.Tensor): Observations from environments.

        Returns:
            The V critic value of observation.
        """
        val = self.net(obs).flatten()
        return val