import torch
import torch.nn.functional as F
from collections import defaultdict, deque
import numpy as np
from .ppo import PPO
from safe_rl_lab.utils.lagrange import PIDLagrange

class PPOLag(PPO):
    def __init__(self, logger, runner, agent, main_optimizer, cost_optimizer, cfg, device="cpu"):
        super().__init__(logger, runner, agent, main_optimizer, cfg, device)
        self.cost_optimizer = cost_optimizer
        self.lagrange = PIDLagrange(
            pid_kp=cfg.algo.k_p,
            pid_ki=cfg.algo.k_i,
            pid_kd=cfg.algo.k_d,
            pid_d_delay=cfg.algo.d_delay,
            pid_delta_d_ema_alpha=cfg.algo.pid_delta_d_ema_alpha,
            pid_delta_p_ema_alpha=cfg.algo.pid_delta_p_ema_alpha,
            sum_norm=cfg.algo.sum_norm,
            diff_norm=cfg.algo.diff_norm,
            penalty_max=cfg.algo.penalty_max,
            lagrangian_multiplier_init=cfg.algo.lagrangian_multiplier_init,
            cost_limit=cfg.algo.cost_limit)

    def update(self, data, rollout_info):
        # --- SECTION 1: Lagrangian Update ---
        if "cost" in rollout_info:
            Jc = rollout_info["cost"]
            self.logger.log({"Jc/real_jc": Jc,}, self.global_step)
        else:
            Jc = data["cost"].mean() * self.cfg.env.max_episode_steps
            self.logger.log({"Jc/est_jc": Jc}, self.global_step)

        self.lagrange.update(Jc)
        cur_lambda = self.lagrange.lagrangian_multiplier

        self.logger.log({
            "cost/lambda": cur_lambda,
            "cost/Jc": Jc,
            "cost/cost_violation": Jc - self.cfg.algo.cost_limit
        }, step=self.global_step)

        c_adv = data['c_adv']
        adv = data["adv"]

        if self.cfg.algo.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            c_adv = (c_adv - c_adv.mean()) / (c_adv.std() + 1e-8)

        adv_total = (adv - cur_lambda * c_adv) / (1 + cur_lambda)
        data['adv'] = adv_total

        # --- SECTION 2: Optimization Loop ---
        batch_size = self.cfg.algo.batch_size
        dataset_size = data["obs"].shape[0]
        b_inds = np.arange(dataset_size)
        update_stats = defaultdict(list)


        for epoch in range(self.cfg.algo.update_epochs):
            epoch_kls = []
            np.random.shuffle(b_inds)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]
                mb = {k: v[mb_inds] for k, v in data.items()}

                #1. Compute all losses
                ppo_loss, cost_loss, stats = self.compute_loss(mb)

                #2. update main agent
                self.optimizer.zero_grad()
                ppo_loss.backward()
                if self.cfg.algo.max_grad_norm is not None and self.cfg.algo.max_grad_norm> 0:
                    torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.cfg.algo.max_grad_norm)
                self.optimizer.step()

                #3. update cost critic
                self.cost_optimizer.zero_grad()
                cost_loss.backward()
                if self.cfg.algo.max_grad_norm is not None and self.cfg.algo.max_grad_norm> 0:
                    torch.nn.utils.clip_grad_norm_(self.agent.cost_critic.parameters(), self.cfg.algo.max_grad_norm)
                self.cost_optimizer.step()

                #collect stats
                epoch_kls.append(stats.get("approx_kl", 0.0))
                for k, v in stats.items():
                    update_stats[k].append(v)

            # KL Based Early Stopping (on epoch level)
            mean_kl = np.mean(epoch_kls)
            if self.cfg.algo.use_kl_early_stopping:
                if mean_kl > self.cfg.algo.early_stopping_target_kl:
                    print("BREAK - ALERT !!! DANGEROUSLY HIGH KL DIVERGENCE")
                    break

        # Final Cleanup
        avg_stats = {k: np.mean(v) for k, v in update_stats.items()}
        return avg_stats


    def compute_loss(self, data):
        ppo_loss, stats = super().compute_loss(data)

        obs = data['obs']
        c_ret = data['c_ret']

        c_val = self.agent.get_cost_value(obs)
        c_loss = torch.nn.functional.mse_loss(c_val, c_ret)
        c_loss = c_loss * 0.5

        stats["cost_value_loss"] = c_loss.item()
        return ppo_loss, c_loss, stats