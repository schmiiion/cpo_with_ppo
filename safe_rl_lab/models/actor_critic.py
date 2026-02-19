# import torch
# import torch.nn as nn
# from torch.distributions import Normal
# from safe_rl_lab.utils.model_utils import build_mlp_network
#
#
#
# class GaussianActor(nn.Module):
#     def __init__(self, obs_dim, act_dim, hidden_sizes, activation, output_activation, weight_initialization_mode):
#         super().__init__()
#         self.actor_net = build_mlp_network(sizes=[obs_dim] + hidden_sizes + [act_dim],
#                                            activation=activation,
#                                            output_activation=output_activation,
#                                            weight_initialization_mode=weight_initialization_mode)
#         self.log_std = nn.Parameter(torch.ones(act_dim) * -0.5, requires_grad=True)
#
#     def forward(self, obs):
#         mu = self.actor_net(obs)
#
#         log_std = torch.clamp(self.log_std, -20, 2)
#         std = torch.exp(log_std)
#         dist = Normal(mu, std)
#
#         return dist

