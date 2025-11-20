from pathlib import Path

import numpy as np
import torch
import wandb
import torch.nn as nn
from gymnasium.vector import VectorEnv


#### OWN
from safe_rl_lab.models.actor_critic import ActorCritic
from safe_rl_lab.models.agent import Agent
from safe_rl_lab.models.sharedBackboneAgent import SharedBackboneAgent
from safe_rl_lab.utils.gae import gae_from_rollout
from safe_rl_lab.runners.single import SingleRunner
from safe_rl_lab.runners.vector import VectorRunner

class PPO:

    def __init__(self, envs, *, model_arch="shared", squash_actions=False, hidden_dim=64,
                 rollout_size=512,
                 gamma=0.99, gae_lambda=0.95, update_epochs=10,
                 minibatch_size=64, lr=3e-4, clip_eps=0.2, vf_coef=0.5,
                 ent_coef=0.01, max_grad_norm=0.5, num_iterations=5000,
                 norm_adv=True, clip_vloss=True, target_kl=0.005, anneal_lr=True,
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
                "optimizer_type": "adam"
            }, allow_val_change=True)

    def train(self):
        if self.model_arch == "separate":
            agent = Agent(self.envs, self.hidden_dim, self.squash_actions)
        elif self.model_arch == "shared":
            agent = SharedBackboneAgent(self.envs, self.hidden_dim, self.squash_actions)

        runner = VectorRunner(self.envs, agent, self.obs_dim, self.act_dim)
        optim = torch.optim.Adam(agent.parameters(), lr=self.lr, eps=1e-5)
        global_steps = 0
        for iteration in range(1, self.num_iterations +1):
            print(f"iteration: {iteration} of {self.num_iterations}")
            if self.anneal_lr:
                frac = 1.0 - (iteration / self.num_iterations)
                lrnow = frac * self.lr
                optim.param_groups[0]['lr'] = lrnow

            buffer, global_steps = runner.run(self.rollout_size, global_steps, agent)
            #store if model is the best yet
            if self.store_model:
                if buffer["ep_rewards_mean"] > self.best_ep_mean:
                    self.best_ep_mean = buffer["ep_rewards_mean"]
                    print('----- storing new model -----')
                    print(buffer["ep_rewards_mean"])
                    self._save_model(iteration, global_steps, agent, optim)

            advantages, returns = self.gae_from_rollout(buffer, self.rollout_size, self.gamma, self.gae_lambda)

            #flatten the batch
            b_obs = buffer["obs"].reshape((-1,) + (self.obs_dim,))
            b_logprobs = buffer["logprob"].reshape(-1)
            b_actions = buffer["act"].reshape((-1,) + (self.act_dim,))
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = buffer["val"].reshape(-1)

            self._ppo_update(agent, optim, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, global_steps)

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            if wandb.run:
                wandb.log({"losses/explained_variance": explained_var}, step=global_steps)


    def _ppo_update(self, agent, optim, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, global_steps):
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        v_loss, pg_loss, entropy_loss = None, None, None

        for epoch in range(self.update_epochs):
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

                # Value loss
                newvalue = newvalue.view(-1)
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

                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                optim.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        if wandb.run:
            wandb.log({
                "charts/learning_rate": optim.param_groups[0]["lr"],
                "losses/value_loss": v_loss.item(),
                "losses/policy_loss": pg_loss.item(),
                "losses/entropy": entropy_loss.item(),
                "losses/old_approx_kl": old_approx_kl.item(),
                "losses/approx_kl": approx_kl.item(),
                "losses/clipfrac": np.mean(clipfracs),
            }, step=global_steps)


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
