from pathlib import Path

import numpy as np
import torch
import wandb
import torch.nn as nn
from gymnasium.vector import VectorEnv
from torch.distributions import kl_divergence, Normal

#### OWN
from safe_rl_lab.models.phasicModel import PhasicValueModel
from safe_rl_lab.runners.vector_runner import VectorRunner
from safe_rl_lab.utils.gae import gae_from_rollout


class _Buffer_B:
    def __init__(self, *, N_pi, rollout_size, num_envs, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs = torch.empty((N_pi, rollout_size * num_envs, obs_dim), dtype=torch.float32)
        self.vtarg = torch.empty((N_pi, rollout_size * num_envs), dtype=torch.float32)
        self.old_mean = torch.empty((N_pi, rollout_size * num_envs, act_dim), dtype=torch.float32)
        self.old_std = torch.empty((N_pi, rollout_size * num_envs, act_dim), dtype=torch.float32)

    def store(self, iteration, b_obs, b_returns, b_pd_means, b_pd_stds):
        self.obs[iteration] = b_obs
        self.vtarg[iteration] = b_returns
        self.old_mean[iteration] = b_pd_means
        self.old_std[iteration] = b_pd_stds

    def get_flattened(self):
        obs = self.obs.reshape((-1,) + (self.obs_dim,))
        v_targets = self.vtarg.view(-1)
        #actions = buffer["acts"] -> analytical KL div used
        old_dist_mean = self.old_mean.reshape((-1,) + (self.act_dim,))
        old_dist_std = self.old_std.reshape((-1,) + (self.act_dim,))
        return obs, v_targets, old_dist_mean, old_dist_std


class PPG:

    def __init__(self, envs, *, hidden_dim=64, rollout_size=512,
                 gamma=0.99, gae_lambda=0.95, update_epochs=10,
                 minibatch_size=64, lr=4e-4, clip_eps=0.2, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.5, num_iterations=5000,
                 norm_adv=True, clip_vloss=True, target_kl=0.005, anneal_lr=False,
                 run_name=None, store_model=False,
                 #PPG params:
                 N_pi =16, E_pi=1, E_v=1, E_aux = 4, aux_minibatch_size=64, beta_clone=1
                 ):
        """lr 4e-4. Compromise between PPG paper (procgen) 5e-4 and tuned Mujoco 3e-4"""

        self.envs = envs
        self.hidden_dim = hidden_dim
        self.obs_dim = envs.observation_space.shape[-1]
        self.act_dim = envs.action_space.shape[-1]
        self.rollout_size = rollout_size
        self.batch_size = rollout_size * envs.num_envs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.lr = lr
        self.anneal_lr = anneal_lr
        self.clip_eps = clip_eps
        self.target_kl = target_kl
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.norm_adv = norm_adv
        self.num_iterations = num_iterations #overall steps: num_iterations*batch_size
        self.clip_vloss = clip_vloss
        #store
        self.run_name = run_name
        self.store_model = store_model
        self.ckpt_dir = Path("checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ep_mean = -float('inf')
        #ppg
        self.N_pi =N_pi
        self.E_pi = E_pi
        self.E_v=E_v
        self.E_aux=E_aux
        self.aux_minibatch_size = aux_minibatch_size
        self.beta_clone = beta_clone


        if wandb.run:
            wandb.config.update({
                "algorithm": "PPG",
                "hidden_dim": hidden_dim,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "rollout_size": rollout_size,
                "batch_size": self.batch_size,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "update_epochs": update_epochs,
                "lr": lr,
                "anneal_lr": anneal_lr,
                "clip_eps": clip_eps,
                "target_kl": target_kl,
                "vf_coef": vf_coef,
                "ent_coef": ent_coef,
                "max_grad_norm": max_grad_norm,
                "norm_adv": norm_adv,
                "num_iterations": num_iterations,
                "clip_vloss": clip_vloss,
                "optimizer_type": "adam",
                #PPG:
                "N_pi": N_pi,
                "E_pi": E_pi,
                "E_v": E_v,
                "E_aux": E_aux,
                "aux_minibatch_size": aux_minibatch_size,
                "beta_clone": beta_clone,
            }, allow_val_change=True)

    def train(self):
        agent = PhasicValueModel(act_dim=self.act_dim, obs_dim=self.obs_dim, hidden_dim=self.hidden_dim)
        optim = torch.optim.Adam(agent.parameters(), lr=self.lr, eps=1e-5)

        runner = VectorRunner(self.envs, agent, self.obs_dim, self.act_dim)
        global_steps = 0

        for phase in range(1, self.num_iterations +1): #phase consists of policy and aux phase
            print(f"phase: {phase} of {self.num_iterations} - marked as purple outer loop in paper")
            buffer_B = _Buffer_B(N_pi=self.N_pi, rollout_size=self.rollout_size,num_envs=self.envs.num_envs, obs_dim=self.obs_dim, act_dim=self.act_dim)

            if self.anneal_lr:
                frac = 1.0 - (phase / self.num_iterations)
                lrnow = frac * self.lr
                optim.param_groups[0]['lr'] = lrnow

            #policy phase
            print('--- start policy phase ---')
            for iteration in range(self.N_pi):
                buffer, global_steps = runner.run(self.rollout_size, global_steps, agent, is_phasic=True)
                if self.store_model: #store if model is the best yet
                    if buffer["ep_rewards_mean"] > self.best_ep_mean:
                        self.best_ep_mean = buffer["ep_rewards_mean"]
                        print(f'----- storing new model  with mean {buffer["ep_rewards_mean"]}-----')
                        self._save_model(iteration, global_steps, agent, optim)

                advantages, returns = gae_from_rollout(buffer, rollout_size=self.rollout_size, gamma=self.gamma, gae_lambda=self.gae_lambda)

                # flatten the batch: [rollout, envs] -> [rollout * envs] -> make sample MORE IID-ish
                b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_old_means, b_old_stds = self._flatten_batch(buffer, advantages, returns)
                buffer_B.store(iteration, b_obs, b_returns, b_old_means, b_old_stds) # in paper: ONLY (s_t, V_t^targ

                for epoch in range(self.E_pi):
                    self._policy_update(agent, optim, b_obs, b_actions, b_logprobs, b_advantages, global_steps)

                for epoch in range(self.E_v):
                    self._value_update(agent, optim, b_obs, b_returns, b_values, global_steps)

            #auxiliary phase
            print('--- start aux phase ---')
            for epoch in range(self.E_aux):
                self._aux_train(buffer_B, agent, optim, global_steps)


    def _policy_update(self, agent, optim, b_obs, b_actions, b_logprobs, b_advantages, global_steps):
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        pg_loss, entropy_loss = None, None

        np.random.shuffle(b_inds)
        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            mb_inds = b_inds[start:end]

            dist, vpredtrue, _ = agent(b_obs[mb_inds])
            newlogprob = dist.log_prob(b_actions[mb_inds]).sum(-1)
            entropy = dist.entropy().sum(-1)
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > self.clip_eps).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if self.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            entropy_loss = entropy.mean()
            l_clip = pg_loss - self.ent_coef * entropy_loss

            optim.zero_grad()
            l_clip.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
            optim.step()

        if wandb.run:
            wandb.log({
                "charts/learning_rate": optim.param_groups[0]["lr"],
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
            }, step=global_steps)


    def _value_update(self, agent, optim, b_obs, b_returns, b_values, global_steps):
        b_inds = np.arange(self.batch_size)
        np.random.shuffle(b_inds)
        v_loss = None

        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            mb_inds = b_inds[start:end]

            newvalue = agent.v(b_obs[mb_inds]).view(-1)
            if self.clip_vloss:
                v_loss_unclipped = (newvalue - b_values[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -self.clip_eps,
                    self.clip_eps,
                )
                ref = b_returns[mb_inds]
                v_loss_clipped = (v_clipped - ref) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            optim.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
            optim.step()

        if wandb.run:
            wandb.log({
                "losses/value_loss": v_loss.item(),
            }, step=global_steps)

    def _aux_train(self, buffer, agent, optim, global_step):
        obs, v_targets, old_dist_mean, old_dist_std = buffer.get_flattened()

        buffer_len = obs.shape[0]
        b_inds = np.arange(buffer_len)
        np.random.shuffle(b_inds)
        for start in range(0, buffer_len, self.aux_minibatch_size):
            end = start + self.aux_minibatch_size
            mb_inds = b_inds[start:end]
            new_pd, _, aux = agent(obs[mb_inds])
            loss_dict = {}
            old_pd = Normal(loc=old_dist_mean[mb_inds], scale=old_dist_std[mb_inds])
            loss_dict["pol_distance"] = kl_divergence(old_pd, new_pd).sum(-1).mean()
            loss_dict.update(agent.compute_aux_loss(aux, v_targets[mb_inds]))

            loss = 0
            for name in loss_dict.keys():
                unscaled_loss = loss_dict[name]
                #uniform weight = 1
                loss += unscaled_loss
                if wandb.run:
                    wandb.log({f"charts/aux_phase_{name}": unscaled_loss}, step=global_step)
            optim.zero_grad()
            loss.backward()
            optim.step()

    def _save_model(self, iteration, global_steps, agent, optim):
        ckpt_path = self.ckpt_dir / f"{self.run_name}_best.pt"
        torch.save({
            "iteration": iteration,
            "steps": global_steps,
            "model_arch": self.model_arch,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        }, ckpt_path)


    def _flatten_batch(self, buffer, advantages, returns):
        b_obs = buffer["obs"].reshape((-1,) + (self.obs_dim,))  # collapse envs dimension -> rollout * envs
        b_logprobs = buffer["logprob"].reshape(-1)
        b_actions = buffer["act"].reshape((-1,) + (self.act_dim,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buffer["val"].reshape(-1)
        b_old_means = buffer["old_pd_mean"].reshape((-1,) + (self.act_dim,))
        b_old_stds = buffer["old_pd_std"].reshape((-1,) + (self.act_dim,))

        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_old_means, b_old_stds