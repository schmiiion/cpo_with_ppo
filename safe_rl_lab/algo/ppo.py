import torch
from safe_rl_lab.algo.policy_gradient import PolicyGradient

class PPO(PolicyGradient):

    def compute_loss(self, data):
        obs = data['obs']
        act = data['act']
        old_logp = data['logp']
        adv = data['adv']
        ret = data['ret']
        old_val = data['val']

        # 1. Forward Pass with gradient tracking for update! - get new sample pds here
        new_logp, entropy, new_val = self.agent.evaluate_actions(obs, act)
        entropy_scalar = entropy.mean()

        # 2. Ratios & KL (Math happens here) -> exp() rausiehen -> Bruch wird zu Subtraktion
        logratio = new_logp - old_logp
        ratio = logratio.exp()

        with torch.no_grad(): #john schulman -> low Variance & non-Negative
            approx_kl = ((ratio - 1) - logratio).mean().item()

        # 3. PPO Loss
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.algo.clip_epsilon, 1.0 + self.cfg.algo.clip_epsilon) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        v_loss_unclipped = (new_val - ret) ** 2
        v_clipped = old_val + torch.clamp(
            new_val - old_val, -self.cfg.algo.clip_epsilon, self.cfg.algo.clip_epsilon,
        )
        v_loss_clipped = (v_clipped - ret) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        total_loss = policy_loss + v_loss - self.cfg.algo.entropy_coef * entropy_scalar

        stats = {
            "policy_loss": policy_loss.item(),
            "v_loss" : v_loss.item(),
            "entropy": entropy_scalar.item(),
            "approx_kl": approx_kl,
        }
        return total_loss, stats