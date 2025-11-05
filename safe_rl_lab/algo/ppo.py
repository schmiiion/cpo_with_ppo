import torch
import wandb
import torch.nn as nn
from gymnasium.vector import VectorEnv


#### OWN
from safe_rl_lab.models.actor_critic import ActorCritic
from safe_rl_lab.runners.single import SingleRunner
from safe_rl_lab.runners.vector import VectorRunner

class PPO:

    def __init__(self, env, model="A2C", hidden_dim=512, rollout_size=256, total_steps=10000,
                 gamma=0.99, lam=0.8, epochs=5, minibatch_size=64,
                 lr=3e-4, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0):
        self.env = env
        self.obs_dim = env.observation_space.shape[-1]
        self.act_dim = env.action_space.shape[-1]
        self.model = ActorCritic(obs_dim=self.obs_dim, act_dim=self.act_dim, hidden_dim=hidden_dim) if model == "A2C" else None
        if self._is_vector_env():
            self.runner = VectorRunner(self.env, self.model, self.obs_dim, self.act_dim)
        else:
            self.runner = SingleRunner(self.env, self.model, self.obs_dim, self.act_dim)
        self.rollout_size = rollout_size
        self.total_steps = total_steps
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)

        if wandb.run:
            wandb.config.update({
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_dim": hidden_dim,
                "rollout_size": rollout_size,
                "gamma": gamma,
                "lam": lam,
                "epochs": epochs,
                "lr": lr,
                "clip_eps": clip_eps,
                "vf_coef": vf_coef,
                "ent_coef": ent_coef,
                "optimizer_type": "adam"
            }, allow_val_change=True)

    def train(self):
        global_steps = 0
        while global_steps < self.total_steps:
            buffer:dict = self.runner.run(self.rollout_size)
            adv, returns = self._gae_from_rollout(buffer)
            assert torch.isfinite(adv).all()
            assert torch.isfinite(returns).all()
            adv_std = adv.std().item()
            if adv_std < 1e-6:
                print("WARNING: advantage std ~ 0. Check done masks and rewards.")
            #normalize advantages per rollout - WHY?
            #adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            self._ppo_update(buffer["obs"], buffer["act"], buffer["logprob"], buffer["val"], adv, returns)

            global_steps += self.rollout_size

            for e in buffer.get("episodes", []):
                if wandb.run:
                    wandb.log({
                        "episode/return": e["return"],
                        "episode/length": e["length"],
                        "charts/total_steps":  global_steps
                    }, step=global_steps)

    def _ppo_update(self, obs, act, logp_old, val_old, adv, ret):
        self.model.train()
        T = obs.shape[0]

        for _ in range(self.epochs):
            idx = torch.randperm(T, device=obs.device)  # why?
            for start in range(0, T, self.minibatch_size):
                mb = idx[start:start + self.minibatch_size] # num_rollouts random numbers(unique) from 0 to T-1
                obs_mb = obs[mb]
                act_mb = act[mb]
                logp0_mb = logp_old[mb].detach()
                adv_mb = adv[mb]
                ret_mb = ret[mb]
                val0_mb = val_old[mb].detach()

                #critic
                v_pred = self.model.forward_critic(obs_mb).squeeze(-1)

                #policy
                logp = self.model.log_prob_of_action(obs_mb, act_mb)

                # NO Entropy bonus so far

                #policy loss (clipped)
                ratio = (logp - logp0_mb).exp()
                pg_unclipped = -adv_mb * ratio
                pg_clipped = -adv_mb * torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps)
                policy_loss = torch.max(pg_unclipped, pg_clipped).mean()

                #value loss - MSE / L2 Norm
                value_loss  = 0.5 * (v_pred - ret_mb).pow(2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef # * entropy

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

                # DIAGNOSTICS
                approx_kl = (logp0_mb - logp).mean().item()
                if wandb.run:
                    wandb.log({
                        "loss/policy": policy_loss.item(),
                        "loss/value": value_loss.item(),
                        "ppo/approx_kl": approx_kl,
                    }, commit=False) # dont advance the global counter yet


    @torch.no_grad()
    def _gae_from_rollout(self, buf):
        """:returns advantages, returns with same shape as buffer["reward"] / ["val"]"""
        rew = buf["rew"]  # [T] or [T, N]
        val = buf["val"]  # [T] or [T, N]
        done = buf["done"]  # [T]  or [T, N] -> (true terminal only)
        v_last = buf["v_last"]  # scalar of [N]

        single = (rew.dim() == 1)

        if single:   # add another dimension for the single env case
            rew = rew.unsqueeze(1)
            val = val.unsqueeze(1)
            done = done.unsqueeze(1)
            v_last = v_last.unsqueeze(0) if v_last.dim() == 1 else v_last

        T, N = rew.shape
        adv = torch.zeros_like(rew)
        next_adv = torch.zeros(N, device=rew.device) #running accumulator
        #next_v = v_last #step init to run backwards  # WHY not needed anymore?
        not_done = (~done).float()

        for t in reversed(range(T)):
            v_next = v_last if t == T-1 else val[t+1]
            delta = rew[t] + self.gamma * not_done[t] * v_next - val[t]
            next_adv = delta + self.gamma * self.lam * not_done[t] * next_adv
            adv[t] = next_adv

        ret = adv + val
        if single:
            adv = adv.squeeze(1)
            ret = ret.squeeze(1)
        return adv, ret

    def _is_vector_env(self):
        if isinstance(self.env, VectorEnv):
            return True
        return False