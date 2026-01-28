import torch
import torch.nn as nn
from safe_rl_lab.utils.model_utils import build_mlp_network


class Critic(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_sizes,
        activation = 'tanh',
        weight_initialization_mode = 'kaiming_uniform',
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
        val = self.net(obs)
        return val