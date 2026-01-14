import torch


@torch.no_grad()
def gae_from_rollout(buf, rollout_size, gamma, gae_lambda):
    """:returns advantages, returns with same shape as buffer["reward"] / ["val"]"""
    with torch.no_grad():
        rewards = buf["rew"]
        vpreds = buf["vpred"]
        dones = buf["done"]

        # Reward bootstrapping
        next_value = buf["v_last"]
        next_done = buf["done_last"]

        # Cost bootstrapping
        costs = buf["cost"]  # Raw signal
        cpreds = buf["cpred"]  # Critic predictions
        next_cpred = buf["cpred_last"]  # Bootstrapped value

        # Output buffers
        adv_r = torch.zeros_like(rewards)
        adv_c = torch.zeros_like(costs)

        lastgaelam_r = 0
        lastgaelam_c = 0

        for t in reversed(range(rollout_size)):
            if t == rollout_size - 1:
                nextnonterminal = 1.0 - next_done
                next_val_r = next_value
                next_val_c = next_cpred
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                next_val_r = vpreds[t + 1]

                next_val_c = cpreds[t + 1]

            # Reward GAE
            delta_r = rewards[t] + gamma * next_val_r * nextnonterminal - vpreds[t]
            adv_r[t] = lastgaelam_r = delta_r + gamma * gae_lambda * nextnonterminal * lastgaelam_r

            # Cost GAE
            delta_c = costs[t] + gamma * next_val_c * nextnonterminal - cpreds[t]
            adv_c[t] = lastgaelam_c = delta_c + gamma * gae_lambda * nextnonterminal * lastgaelam_c

        returns_r = adv_r + vpreds
        returns_c = adv_c + cpreds

        return adv_r, returns_r, adv_c, returns_c