import torch
from .ppo import PPO
from safe_rl_lab.utils.lagrange import PIDLagrange

class PPOLag(PPO):
    def __init__(self, logger, runner, agent, cfg, device="cpu"):
        super().__init__(logger, runner, agent, cfg, device)
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


    def _update(self, rollout_info):
        data = self.buffer.get()

        # --- SECTION 1: Lagrangian Update ---
        if "scaled_cost" in rollout_info:
            Jc = rollout_info["scaled_cost"]
        else:
            Jc = data["cost"].mean() * self.cfg.env.max_episode_steps

        self.lagrange.update(Jc)

        self.logger.log({
            "cost/lambda": self.lagrange.lagrangian_multiplier,
            "cost/Jc": Jc,
            "cost/cost_violation": Jc - self.cfg.algo.cost_limit
        }, step=self.global_step)

        update_state = super()._update(rollout_info)
        return update_state


    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function of reward to update policy network.
        """
        penalty = self.lagrange.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)








