import torch
import torch.nn as nn

class PPO:

    def __init__(self, obs_dim=56, act_dim=8):
        self.backbone_model = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(512, act_dim)
        self.value_head = nn.Linear(512, 1)

        self.optimizer = torch.optim.AdamW(self.backbone_model.parameters(), lr=1e-3)


    def train(self, env):
        obs, info = env.reset()

        terminated, truncated = False, False  # terminated is success/failure do to the env.; Truncated -> stopped for external reasons
        ep_return, ep_cost = 0, 0

        for _ in range(1000):
            assert env.observation_space.contains(obs)
            obs = torch.from_numpy(obs).float()
            #x = self.policy_head(self.backbone_model(obs))
            act = env.action_space.sample()
            assert env.action_space.contains(act)

            obs, reward, cost, terminated, truncated, info = env.step(act)

            #self.train_a2c(obs)

            ep_return += reward
            ep_cost += cost
            if terminated or truncated:
                observation, info = env.reset()

        env.close()



    def train_a2c(self, obs):
        hidden_state = self.backbone_model(obs)
        policy_logits = self.policy_head(hidden_state)
        value_pred = self.value_head(hidden_state)

        self.optimizer.zero_grad()
        # loss =


    class ActorCritic(nn.Module):
        def __init__(self, obs_dim, act_dim, hidden_dim=512):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
            )
            self.policy = nn.Linear(hidden_dim, act_dim)
            self.value = nn.Linear(hidden_dim, 1)

        def forward(self, obs):
            hidden_s = self.backbone(obs)
            policy_logits = self.policy(hidden_s)
            value_pred = self.value(hidden_s)