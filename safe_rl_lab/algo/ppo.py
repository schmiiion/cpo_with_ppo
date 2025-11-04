import torch
# from wandb.apis.importers import wandb
import wandb
import torch.nn as nn

#### OWN
from safe_rl_lab.models.actor_critic import ActorCritic
from safe_rl_lab.runners.single import SingleRunner


class PPO:

    def __init__(self, env, model="A2C", runner_type="single", obs_dim=56, act_dim=8, hidden_dim=512,
                 rollout_size=128, total_steps=10000, gamma=0.99, lam=0.8, epochs=5, minibatch_size=32,
                 lr=3e-4, clip_eps=0.2, vf_coef=0.5, ent_coef=0.0):
        self.model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=hidden_dim) if model == "A2C" else None
        self.runner = SingleRunner(env, self.model, obs_dim, act_dim) if runner_type == "single" else None
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

        wandb.config.update({
            "obs_dim": obs_dim,
            "act_dim": act_dim,
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
            #normalize advantages per rollout - WHY?
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            self._ppo_update(buffer["obs"], buffer["act"], buffer["logprob"], buffer["val"], adv, returns)

            global_steps += self.rollout_size
            wandb.log({
                "reward/mean": buffer["rew"].mean(axis=0).item(),
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
                logp0_mb = logp_old[mb]
                adv_mb = adv[mb]
                ret_mb = ret[mb]
                val0_mb = val_old[mb]

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
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_eps)
                self.optim.step()

                # DIAGNOSTICS
                approx_kl = (logp0_mb - logp).mean().item()
                wandb.log({
                    "loss/policy": policy_loss.item(),
                    "loss/value": value_loss.item(),
                    "ppo/approx_kl": approx_kl,
                }, commit=False) # dont advance the global counter yet


    @torch.no_grad()
    def _gae_from_rollout(self, buf):
        # buf: {"rew":[T], "val":[T], "done":[T], "v_last": scalar}
        rew = buf["rew"]  # float32 [T]
        val = buf["val"]  # float32 [T]
        done = buf["done"]  # bool    [T] (true terminal only)
        v_last = buf["v_last"]  # float32 []

        T = rew.shape[0]
        adv = torch.zeros_like(rew)
        next_adv = torch.zeros((), device=rew.device) #running accumulator
        next_v = v_last #step init to run backwards
        not_done = (~done).float()

        for t in reversed(range(T)):
            delta = rew[t] + self.gamma * not_done[t] * next_v - val[t]
            next_adv = delta + self.gamma * self.lam * not_done[t] * next_adv
            adv[t] = next_adv
            next_v = val[t]

        ret = adv + val
        return adv, ret
