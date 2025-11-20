from safe_rl_lab.models.abstractBaseClasses import PhasicModel
import numpy as np
import torch.nn as nn
import torch
from torch.distributions import Normal

def _layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

def _get_env_dims(env):
    # Handle both single and vectorized envs
    if hasattr(env, "single_observation_space"):
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        obs_space = env.observation_space
        act_space = env.action_space

    obs_dim = int(np.prod(obs_space.shape))
    act_dim = int(np.prod(act_space.shape))
    return obs_dim, act_dim

class PhasicVanillaModel(PhasicModel):
    def __init__(self, envs, hidden_dim):
        super().__init__()
        obs_dim, act_dim = _get_env_dims(envs)

        self.policy_backbone = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.policy_mean = _layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.policy_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.policy_value_head = nn.Linear(hidden_dim, 1)

        self.value_network = nn.Sequential(
            _layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            _layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, ob) -> "pd, vpred, aux":

        h = self.policy_backbone(ob)
        action_mean = self.policy_mean(h)
        action_logstd = self.policy_logstd.expand_as(action_mean)
        action_std = action_logstd.exp()
        dist = Normal(action_mean, action_std)

        vpredtrue = self.value_network(ob)
        vpredaux = self.policy_value_head(h)

        aux = {"vpredtrue": vpredtrue, "vpredaux": vpredaux}

        return dist, vpredtrue, aux

    def aux_keys(self):
        return ["vtarg"]

    def compute_aux_loss(self, aux, mb_targets):
        """I'd implement this loss outpside of the model - the clip loss is also computed outside. But i stick to the PPG repo"""
        return {
            "vf_true": 0.5 * ((aux["vpredtrue"] - mb_targets) ** 2).mean(),
            "vf_aux": 0.5 * ((aux["vpredaux"] - mb_targets) ** 2).mean(),
        }

