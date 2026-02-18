import torch
from safe_rl_lab.algo.policy_gradient import PolicyGradient

class PPO(PolicyGradient):

    def _loss_pi(self, obs, act, logp, adv):

        # 1. Forward Pass with gradient tracking for update! - get new sample pds here
        new_logp, entropy = self._actor_critic.evaluate_actions(obs, act)

        # 2. Ratios & KL (Math happens here) -> exp() rausiehen -> Bruch wird zu Subtraktion
        ratio = torch.exp(new_logp - logp)
        ratio_clipped = torch.clamp(ratio, 1.0 - self.cfg.algo.clip_epsilon, 1.0 + self.cfg.algo.clip_epsilon)

        # with torch.no_grad(): #john schulman -> low Variance & non-Negative
        #     approx_kl = ((ratio - 1) - ratio).mean().item()

        loss = -torch.min(ratio*adv, ratio_clipped * adv).mean()
        loss -= self.cfg.algo.entropy_coef * entropy.mean()

        stats = {
            "policy_loss": loss.mean().item(),
            # "ratio": ratio,
            "entropy": entropy.mean().item(),
        }
        return loss, stats

    # v_loss_unclipped = (new_val - ret) ** 2
    # v_clipped = old_val + torch.clamp(
    #     new_val - old_val, -self.cfg.algo.clip_epsilon, self.cfg.algo.clip_epsilon,
    # )
    # v_loss_clipped = (v_clipped - ret) ** 2
    # v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()