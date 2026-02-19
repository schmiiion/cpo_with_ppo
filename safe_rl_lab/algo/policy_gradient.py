from abc import ABC, abstractmethod
from collections import defaultdict
from safe_rl_lab.algo.base_algo import BaseAlgo
from safe_rl_lab.utils.rollout_buffer import RolloutBuffer
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

class PolicyGradient(BaseAlgo, ABC):
    """
    agent is a container to hold the policy and value function. Either shared or disjoint.
    """
    def __init__(self, logger, runner, a2c, cfg, device="cpu"):
        super().__init__(runner, logger, cfg, device)
        self._actor_critic = a2c
        self.buffer = None

    def learn(self):
        """
        Main training loop:
        1. Sample buffer from environment
        2. Compute GAE
        3. Compute policy gradient (Loss?)
        4. Update policy
        """
        num_updates = self.cfg.total_steps // self.cfg.algo.rollout_size

        self.buffer = RolloutBuffer(
            num_steps=self.cfg.algo.rollout_size,
            num_envs=self.runner.num_envs,
            obs_shape=self.runner.obs_shape,
            act_shape=self.runner.act_shape,
            standardized_adv_r=self.cfg.algo.normalize_adv_r,
            standardized_adv_c=self.cfg.algo.normalize_adv_c,
            device=self.device,
            use_cost=self.cfg.algo.use_cost,
        )

        #Generic update Loop for all Policy Gradient Algorithms
        for update in range(1, num_updates + 1):
            print(f"update iteration {update} of {num_updates +1}")

            if update % self.cfg.sps_logging_interval == 0:
                super()._log_sps(self.global_step)

            self._anneal_lr()

            # 1. Sample Buffer and return bootstrap values
            global_step, rollout_info, last_val, last_done, last_cval = self.runner.run(self._actor_critic, self.buffer, self.global_step)
            self.global_step = global_step
            self._process_episodic_stats(rollout_info)

            #  2. Compute GAE (inside Buffer)
            self.buffer.compute_gae(last_val, last_done, self.cfg.algo.gae.gamma, self.cfg.algo.gae.lam, last_cval)

            # UPDATE - Template Method
            update_state = self._update(rollout_info)
            self.logger.log(metrics=update_state, step=self.global_step, prefix="train")


    def _update(self, rollout_info):
        """
        Standard PPO-style update loop.
        Can be used by PPO, PPO-Lag, and PPG (Policy Phase).
        """
        update_stats = defaultdict(list)

        data = self.buffer.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )

        original_obs = data["obs"]
        old_distribution = self._actor_critic.get_distribution(obs)

        update_counts = 0
        final_kl = 0.0

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self.cfg.algo.batch_size,
            shuffle=True,
        )

        for epoch in range(self.cfg.algo.update_epochs):

            for (obs, act, logp, target_value_r, target_value_c, adv_r, adv_c,) in dataloader:

                #SEPARATE OPTIMIZATION OF THREE NETWORKS
                if self.cfg.algo.a2c_architecture == "separate":
                    #update reward critic and store loss
                    v_loss_item = self._update_reward_critic(obs, target_value_r)
                    update_stats["v_loss"].append(v_loss_item)

                    #optionally update cost critic and store loss
                    if self.cfg.algo.use_cost:
                        c_loss_item = self._update_cost_critic(obs, target_value_c)
                        update_stats["c_loss"].append(c_loss_item)

                    #update actor and store loss
                    pi_loss_item, loss_update_stats = self._update_actor(obs, act, logp, adv_r, adv_c)
                    update_stats["pi_loss"].append(pi_loss_item)
                    for key, value in loss_update_stats.items():
                        update_stats[key].append(value)


                # JOINT OPTIMIZATION OF ACTOR/ REWARD CRITIC / COST CRITIC
                else:
                    joint_update_stats = self._joint_update(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c)
                    for key, value in joint_update_stats.items():
                        update_stats[key].append(value)


            new_distribution = self._actor_critic.get_distribution(original_obs)
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )

            final_kl = kl.item()
            update_counts += 1

            if self.cfg.algo.kl_early_stop and kl.item() > self.cfg.algo.target_kl:
                break

        #aggregate stats after all epochs
        avg_stats = {k: np.mean(v) for k, v in update_stats.items()}
        avg_stats["final_kl"] = final_kl
        avg_stats["update_counts"] = update_counts
        return avg_stats

    def _update_actor(self, obs: torch.Tensor, act: torch.Tensor, logp: torch.Tensor, adv_r: torch.Tensor, adv_c: torch.Tensor):
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss, stats = self._loss_pi(obs, act, logp, adv)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()

        if self.cfg.algo.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self.cfg.algo.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()
        return loss.item(), stats

    def _update_reward_critic(self, obs, target_value_r):
        self._actor_critic.reward_critic_optimizer.zero_grad()
        #needs predictions
        predictions = self._actor_critic.reward_critic(obs)
        loss = nn.functional.mse_loss(predictions, target_value_r)

        if self.cfg.algo.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self.cfg.algo.critic_norm_coef

        loss.backward()

        if self.cfg.algo.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self.cfg.algo.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()

        return loss.item()


    def _update_cost_critic(self, obs: torch.Tensor, target_value_c: torch.Tensor):
        self._actor_critic.cost_critic_optimizer.zero_grad()
        predictions = self._actor_critic.cost_critic(obs)
        loss = nn.functional.mse_loss(predictions, target_value_c)

        if self.cfg.algo.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self.cfg.algo_cfgs.critic_norm_coef

        loss.backward()

        if self.cfg.algo.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self.cfg.algo.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        return loss.item()


    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        return adv_r

    def _joint_update(self, obs, act, logp, target_value_r, target_value_c, adv_r, adv_c):
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        self._actor_critic.optimizer.zero_grad()

        loss, update_stats = self._joint_loss(obs, act, logp, adv, target_value_r, target_value_c)
        loss.backward()

        if self.cfg.algo.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.parameters(),
                self.cfg.algo.max_grad_norm,
            )

        self._actor_critic.optimizer.step()
        return update_stats

    @abstractmethod
    def _loss_pi(self, obs, act, logp, adv):
        """
        Must be implemented by Child (PPO, PPO-Lag).
        """
        pass

    def _anneal_lr(self):
        """Maybe in optimizer wrapper?"""
        pass

    @abstractmethod
    def _joint_loss(self, obs, act, logp, adv, target_value_r, target_value_c):
        pass
