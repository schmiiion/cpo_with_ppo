from pathlib import Path

import numpy as np
from collections import deque
import time
import torch
import wandb
import torch.nn as nn
from gymnasium.vector import VectorEnv


#### OWN
from safe_rl_lab.models.actor_critic import ActorCritic
from safe_rl_lab.models.actor_critic_disjoint import ActorCriticDisjoint
from safe_rl_lab.models.cost_critic import CostCritic
from safe_rl_lab.models.sharedBackboneAgent import SharedBackboneAgent
from safe_rl_lab.utils.gae import gae_from_rollout
from safe_rl_lab.utils.lagrange import Lagrange
from safe_rl_lab.runners.vector_runner import VectorRunner

class PPOLag:

    def __init__(self, envs, *, model_arch="shared", squash_actions=False, hidden_dim=64,
                 rollout_size=512,
                 gamma=0.99, gae_lambda=0.95, update_epochs=10,
                 minibatch_size=64, lr=3e-4, clip_eps=0.2, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.5, num_iterations=10000,
                 norm_adv=True, clip_vloss=True, target_kl=0.005, anneal_lr=True,
                 cost_limit=50, lambda_lr = 5e-2,
                 run_name=None, store_model=False):
        self.envs = envs
        self.model_arch = model_arch
        self.squash_actions = squash_actions
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

        self.run_name = run_name
        self.store_model = store_model
        self.ckpt_dir = Path("checkpoints")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.best_ep_mean = -float('inf')

        #wall clock time:
        self.sps_logging_interval = 10
        self.sps_window = deque(maxlen=10)  # Keep last 10 measurements
        self.last_log_time = time.time()
        self.last_global_step = 0

        self.cost_limit = cost_limit
        self.lagrange = None
        self.lambda_lr = lambda_lr

        if wandb.run:
            wandb.config.update({
                "model_arch": model_arch,
                "squash_actions": squash_actions,
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
                "cost_limit": cost_limit,
            }, allow_val_change=True)

    def train(self):
        if self.model_arch == "shared":
            agent = ActorCritic(obs_dim=self.obs_dim, act_dim=self.act_dim, hidden_dim=self.hidden_dim)
        else:
            raise RuntimeError("Invalid agent architecture")
        cost_critic = CostCritic(self.obs_dim, hidden_sizes=[self.hidden_dim, self.hidden_dim], activation='tanh')

        optim = torch.optim.Adam(agent.parameters(), lr=self.lr, eps=1e-5)
        cost_critic_optim = torch.optim.Adam(cost_critic.parameters(), lr=self.lr, eps=1e-5)
        self.lagrange = Lagrange(cost_limit=self.cost_limit, lagrangian_multiplier_init=0.001, lambda_lr=self.lambda_lr)

        runner = VectorRunner(self.envs, agent, self.obs_dim, self.act_dim, cost_critic)

        global_steps, last_global_step = 0, 0
        for iteration in range(1, self.num_iterations +1):
            print(f"iteration: {iteration} of {self.num_iterations}")

            if iteration % self.sps_logging_interval == 0 and iteration > 0:
                self._comp_and_log_wall_time(global_steps)

            if self.anneal_lr:
                frac = 1.0 - (iteration / self.num_iterations)
                lrnow = frac * self.lr
                optim.param_groups[0]['lr'] = lrnow

            buffer, global_steps = runner.run(self.rollout_size, global_steps, agent)
            #store if model is the best yet
            if self.store_model:
                if buffer["ep_rewards_mean"] > self.best_ep_mean:
                    self.best_ep_mean = buffer["ep_rewards_mean"]
                    print(f'----- storing new model with mean reward {buffer["ep_rewards_mean"]} and cost {buffer["ep_cost_mean"]}')

                    self._save_model(iteration, global_steps, agent, optim)

            adv_r, ret_r, adv_c, ret_c = gae_from_rollout(buffer, self.rollout_size, self.gamma, self.gae_lambda)

            Jc = buffer["ep_cost_mean"]
            if Jc == 0.0 and buffer["cost"].sum() > 0: # Fallback if no episode finished
                print('#### Jc fallback used!')
                # Rough estimate: (avg_step_cost) * (max_ep_len)
                Jc = buffer["cost"].mean() * 1000  # assuming 1000 is max_ep_len

            self.lagrange.update_lagrange_multiplier(Jc)
            lambda_val = self.lagrange.lagrangian_multiplier.item()
            adv_total = (adv_r - lambda_val * adv_c) / (1 + lambda_val)

            if self.norm_adv:
                adv_total = (adv_total - adv_total.mean()) / (adv_total.std() + 1e-8)

            #Flatten the batch
            b_obs = buffer["obs"].reshape((-1,) + (self.obs_dim,))
            b_actions = buffer["act"].reshape((-1,) + (self.act_dim,))
            b_logprobs = buffer["logprob"].reshape(-1)
            b_adv_total = adv_total.reshape(-1)
            #reward data:
            b_ret_r = ret_r.reshape(-1)
            b_vpred = buffer["vpred"].reshape(-1)
            #cost data:
            b_ret_c = ret_c.reshape(-1)
            b_cpred = buffer["cpred"].reshape(-1)

            self._ppo_lag_update(agent, optim, cost_critic, cost_critic_optim, b_obs, b_actions, b_logprobs,
                                 b_adv_total, b_ret_r, b_vpred, b_ret_c, b_cpred, global_steps)

            y_pred, y_true = b_vpred.cpu().numpy(), b_ret_r.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            if wandb.run:
                wandb.log({"losses/explained_variance": explained_var}, step=global_steps)


    def _ppo_lag_update(self, agent, optim, cost_critic, cost_critic_optim, b_obs, b_actions, b_logprobs, b_adv_total, b_ret_r, b_vpreds, b_ret_c, b_cpreds, global_steps):
        b_inds = np.arange(self.batch_size)
        clipfracs = []

        v_loss_r_epoch, v_loss_c_epoch, pg_loss_epoch, entropy_loss_epoch = [], [], [], []

        for epoch in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                # ---------------------------------------------------------
                # 1. Forward Passes
                #Agent (Actor + Reward Critic)
                pdf, vpred_r, _ = agent(b_obs[mb_inds])
                #Cost Critic
                vpred_c = cost_critic(b_obs[mb_inds]).flatten()

                # ---------------------------------------------------------
                # 2. Policy Loss (Actor)
                new_log_probs = pdf.log_prob(b_actions[mb_inds]).sum(dim=-1)
                old_log_probs = b_logprobs[mb_inds]
                logratio = new_log_probs - old_log_probs
                ratio = logratio.exp()

                with torch.no_grad(): # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_eps).float().mean().item()]

                mb_advantages = b_adv_total[mb_inds]
                # if self.norm_adv:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # ---------------------------------------------------------
                # 3. Value Loss (Reward Critic)
                newvalue_r = vpred_r.view(-1) #flatten to make compatible
                v_loss_unclipped = (newvalue_r - b_ret_r[mb_inds]) ** 2
                v_clipped = b_vpreds[mb_inds] + torch.clamp(
                    newvalue_r - b_vpreds[mb_inds], -self.clip_eps, self.clip_eps,
                )
                v_loss_clipped = (v_clipped - b_ret_r[mb_inds]) ** 2
                v_loss_r = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # ---------------------------------------------------------
                # 4. Value Loss (Cost Critic)
                v_loss_unclipped_c = (vpred_c - b_ret_c[mb_inds]) ** 2
                v_clipped_c = b_cpreds[mb_inds] + torch.clamp(
                    vpred_c - b_cpreds[mb_inds], -self.clip_eps, self.clip_eps
                )
                v_loss_clipped_c = (v_clipped_c - b_ret_c[mb_inds]) ** 2
                v_loss_c = 0.5 * torch.max(v_loss_unclipped_c, v_loss_clipped_c).mean()

                # ---------------------------------------------------------
                # 5. Optimization Steps
                entropy_loss = pdf.entropy().mean()

                #1. Update Agent
                loss_agent = pg_loss - self.ent_coef * entropy_loss + v_loss_r * self.vf_coef

                optim.zero_grad()
                loss_agent.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                optim.step()

                #2. Update Cost Critic
                cost_critic_optim.zero_grad()
                v_loss_c.backward()
                nn.utils.clip_grad_norm_(cost_critic.parameters(), self.max_grad_norm)
                cost_critic_optim.step()

                #Logging
                v_loss_r_epoch.append(v_loss_r.item())
                v_loss_c_epoch.append(v_loss_c.item())
                pg_loss_epoch.append(pg_loss.item())
                entropy_loss_epoch.append(entropy_loss.item())

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        if wandb.run:
            wandb.log({
                "losses/value_loss_reward": np.mean(v_loss_r_epoch),
                "losses/value_loss_cost": np.mean(v_loss_c_epoch),
                "losses/policy_loss": np.mean(pg_loss_epoch),
                "losses/entropy": np.mean(entropy_loss_epoch),
                "losses/approx_kl": approx_kl.item(),
            }, step=global_steps)


    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        PPOLag uses the following surrogate loss:

        .. math::
                L = \frac{1}{1 + \lambda} [
                    A^{R}_{\pi_{\theta}} (s, a)
                    - \lambda A^C_{\pi_{\theta}} (s, a)
                ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)

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

    def _comp_and_log_wall_time(self, global_steps):
        current_time = time.time()
        elapsed_time = current_time - self.last_log_time

        steps_diff = global_steps - self.last_global_step
        self.sps_window.append(steps_diff / elapsed_time)
        avg_sps = np.mean(self.sps_window)
        #print(f"avg sps: {avg_sps:.2f}")

        if wandb.run:
            wandb.log({"charts/SPS_mean": avg_sps}, step=global_steps)

        # Reset trackers
        self.last_log_time = current_time
        self.last_global_step = global_steps
