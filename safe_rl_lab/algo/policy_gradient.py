from abc import ABC, abstractmethod
from collections import defaultdict
from safe_rl_lab.algo.base_algo import BaseAlgo
from safe_rl_lab.utils.rollout_buffer import RolloutBuffer
import numpy as np
import torch

class PolicyGradient(BaseAlgo, ABC):
    """
    agent is a container to hold the policy and value function. Either shared or disjoint.
    """
    def __init__(self, logger, runner, agent, optimizer, cfg, device="cpu"):
        super().__init__(runner, logger, cfg, device)
        self.agent = agent
        self.optimizer = optimizer
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
            global_step, rollout_info, last_val, last_done, last_cval = self.runner.run(self.agent, self.buffer, self.global_step)
            self.global_step = global_step
            # log stats and safe a new best modelx
            self._process_episodic_stats(rollout_info)
            # self._safe_if_best(global_step)

            #  2. Compute GAE (inside Buffer)
            self.buffer.compute_gae(last_val, last_done, self.cfg.algo.gae.gamma, self.cfg.algo.gae.lam, last_cval)

            # UPDATE - Template Method
            buffer_data = self.buffer.get()
            update_state = self.update(buffer_data, rollout_info)
            self.logger.log(metrics=update_state, step=self.global_step, prefix="train")


    def update(self, data, rollout_info):
        """
        Standard PPO-style update loop.
        Can be used by PPO, PPO-Lag, and PPG (Policy Phase).
        """
        batch_size = self.cfg.algo.batch_size
        dataset_size = data["obs"].shape[0]
        b_inds = np.arange(dataset_size)
        update_stats = defaultdict(list)

        if self.cfg.algo.normalize_adv:
            data['adv'] = (data['adv'] - data['adv'].mean()) / (data['adv'].std() + 1e-8)

        for epoch in range(self.cfg.algo.update_epochs):
            epoch_kls = []
            np.random.shuffle(b_inds)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]
                mb = {k: v[mb_inds] for k, v in data.items()}

                loss, stats = self.compute_loss(mb)

                self.optimizer.zero_grad()
                loss.backward()
                if self.cfg.algo.max_grad_norm is not None and self.cfg.algo.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.algo.max_grad_norm)
                self.optimizer.step()

                #collect stats
                epoch_kls.append(stats.get("approx_kl", 0.0))
                for k, v in stats.items():
                    update_stats[k].append(v)

            # KL Based Early Stopping (on epoch level)
            mean_kl = np.mean(epoch_kls)
            if self.cfg.algo.use_kl_early_stopping:
                if mean_kl > self.cfg.algo.early_stopping_target_kl:
                    break

        #aggregate stats after all epochs
        avg_stats = {k: np.mean(v) for k, v in update_stats.items()}

        # Hook for PPG / Logging / Cleaning up
        return self.on_update_end(avg_stats)


    def on_update_end(self, data):
        return data

    @abstractmethod
    def compute_loss(self, batch):
        """
        Must be implemented by Child (PPO, PPO-Lag).
        Returns: (torch.Tensor scalar_loss, dict stats_for_logging)
        """
        pass

    def _anneal_lr(self):
        """Maybe in optimizer wrapper?"""
        pass
