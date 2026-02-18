import torch
from .ppo import PPO
from safe_rl_lab.utils.lagrange import PIDLagrange
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

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

    def _joint_update(self, obs, act, logp, target_value_r, target_value_c, adv_r, adv_c):
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        self._actor_critic.optimizer.zero_grad()

        # 1. Forward Pass with gradient tracking for update!
        new_logp, entropy, pred_value_r, pred_value_c = self._actor_critic.evaluate_actions(obs, act)

        # Policy Gradient Loss
        ratio = torch.exp(new_logp - logp)
        ratio_clipped = torch.clamp(ratio, 1.0 - self.cfg.algo.clip_epsilon, 1.0 + self.cfg.algo.clip_epsilon)
        loss = -torch.min(ratio*adv, ratio_clipped * adv).mean()

        #Reward Value Loss
        value_rew_loss = nn.functional.mse_loss(pred_value_r, target_value_r)
        loss -= self.cfg.algo.beta_value_reward * value_rew_loss

        #Cost Value Loss
        value_cost_loss = nn.functional.mse_loss(pred_value_r, target_value_r)
        loss -= self.cfg.algo.beta_value_cost * value_cost_loss

        loss.backward()

        if self.cfg.algo.use_max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.optimizer.parameters(),
                self.cfg.algo.max_grad_norm,
            )
        self._actor_critic.optimizer.step()






