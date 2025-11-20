from pathlib import Path

import numpy as np
import torch
import wandb
import torch.nn as nn
from gymnasium.vector import VectorEnv
from torch.distributions import kl_divergence, Normal

#### OWN
from safe_rl_lab.models.phasicModel import PhasicVanillaModel
from safe_rl_lab.runners.vector import VectorRunner
from safe_rl_lab.utils.gae import gae_from_rollout


class PPG:

    def __init__(self, envs, *, hidden_dim=64, rollout_size=512,
                 gamma=0.99, gae_lambda=0.95, update_epochs=10,
                 minibatch_size=64, lr=3e-4, clip_eps=0.2, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.5, num_iterations=5000,
                 norm_adv=True, clip_vloss=True, target_kl=0.005, anneal_lr=True,
                 run_name=None, store_model=False,
                 #PPG params:
                 N_pi =16, E_pi=1, E_v=6, E_aux = 6, aux_minibatch_size=10, beta_clone=0.5
                 ):

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
                "optimizer_type": "adam"
            }, allow_val_change=True)

    def train(self):
        agent = PhasicVanillaModel(envs=self.envs, hidden_dim=self.hidden_dim)
        optim = torch.optim.Adam(agent.parameters(), lr=self.lr, eps=1e-5)

        runner = VectorRunner(self.envs, agent, self.obs_dim, self.act_dim)
        global_steps = 0

        #phase consists of policy and aux phase
        for phase in range(1, self.num_iterations +1):
            print(f"phase: {phase} of {self.num_iterations}")
            buffer_B = self._create_buffer_B()

            if self.anneal_lr:
                frac = 1.0 - (phase / self.num_iterations)
                lrnow = frac * self.lr
                optim.param_groups[0]['lr'] = lrnow

            #policy phase
            for iteration in range(self.N_pi):
                buffer, global_steps = runner.run(self.rollout_size, global_steps, agent)
                #store if model is the best yet
                if self.store_model:
                    if buffer["ep_rewards_mean"] > self.best_ep_mean:
                        self.best_ep_mean = buffer["ep_rewards_mean"]
                        print('----- storing new model -----')
                        print(buffer["ep_rewards_mean"])
                        self._save_model(iteration, global_steps, agent, optim)

                advantages, returns = gae_from_rollout(buffer, rollout_size=self.rollout_size, gamma=self.gamma, gae_lambda=self.gae_lambda)

                # flatten the batch: [rollout, envs] -> [rollout * envs]
                b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = self._flatten_batch(buffer, advantages, returns, buffer_B)
                #add to buffer
                buffer_B["old_mean"].append(b_advantages)
                buffer_B["old_logstd"].append(b_values)

                for epoch in range(self.E_pi):
                    self._policy_update(agent, optim, b_obs, b_actions, b_logprobs, b_advantages, b_returns)

                for epoch in range(self.E_v):
                    self._value_update(agent, optim, b_obs, b_actions, b_logprobs, b_returns)

            # y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            # var_y = np.var(y_true)
            # explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            # if wandb.run:
            #     wandb.log({"losses/explained_variance": explained_var}, step=global_steps)

            #auxiliary phase
            for epoch in range(self.E_aux):
                self._aux_update(buffer_B, self.agent, optim)
                self._value_update()


    def _policy_update(self, agent, optim, b_obs, b_actions, b_logprobs, b_advantages, global_steps):
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        pg_loss, entropy_loss = None, None

        np.random.shuffle(b_inds)
        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
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
        clipfracs = []
        v_loss = None

        np.random.shuffle(b_inds)
        for start in range(0, self.batch_size, self.minibatch_size):
            end = start + self.minibatch_size
            mb_inds = b_inds[start:end]

            newvalue = agent.v(b_obs[mb_inds]).view(-1)
            if self.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -self.clip_eps,
                    self.clip_eps,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
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

    def _aux_update(self, buffer, agent):
        obs = buffer["obs"]
        v_targets = buffer["vtarg"]
        #actions = buffer["acts"] -> analytical KL div used
        old_dist_mean = buffer["old_pd_mean"]
        old_dist_std = buffer["old_pd_std"]

        buffer_size = len(obs)
        b_inds = np.arange(buffer_size)
        np.random.shuffle(b_inds)
        for start in range(0, buffer_size, self.aux_minibatch_size):
            end = start + self.aux_minibatch_size
            mb_inds = b_inds[start:end]
            new_dist, vpredtrue, aux = agent(obs[mb_inds])
            mb_targets = v_targets[mb_inds]
            L_aux = agent.compute_aux_loss(aux, mb_targets)

            old_dist = Normal(old_dist_mean, old_dist_std)
            kl_div = kl_divergence(old_dist, new_dist)

            L_joint = L_aux + self.beta_clone * kl_div
            L_joint.backward()







    def _is_vector_env(self):
        if isinstance(self.envs, VectorEnv):
            return True
        return False

    def _save_model(self, iteration, global_steps, agent, optim):
        ckpt_path = self.ckpt_dir / f"{self.run_name}_best.pt"
        torch.save({
            "iteration": iteration,
            "steps": global_steps,
            "model_arch": self.model_arch,
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
        }, ckpt_path)

    def _create_buffer_B(self):
        return {
            "obs": [],
            "actions": [],
            "old_logprob": [],
            "vtarg": [],
            "old_mean": [],
            "old_logstd": []
        }

    def _flatten_batch(self, buffer, advantages, returns, buffer_B=None):
        b_obs = buffer["obs"].reshape((-1,) + (self.obs_dim,))  # collapse envs dimension -> rollout * envs
        b_logprobs = buffer["logprob"].reshape(-1)
        b_actions = buffer["act"].reshape((-1,) + (self.act_dim,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buffer["val"].reshape(-1)
        if buffer_B is not None:
            buffer_B["obs"].append(b_obs)
            buffer_B["actions"].append(b_actions)
            buffer_B["old_logprob"].append(b_logprobs)
            buffer_B["vtarg"].append(b_returns)

        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values
