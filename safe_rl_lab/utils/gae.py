import torch

@torch.no_grad()
def gae_from_rollout(buf, rollout_size, gamma, gae_lambda):
    """:returns advantages, returns with same shape as buffer["reward"] / ["val"]"""
    with torch.no_grad():
        rewards = buf["rew"]  # [T] or [T, N]
        values = buf["val"]  # [T] or [T, N]
        dones = buf["done"]  # [T]  or [T, N] -> (true terminal only)
        next_value = buf["v_last"]  # scalar of [N]
        next_done = buf["done_last"]
        advantages = torch.zeros_like(rewards)
        lastgaelam = 0

        # single = (rew.dim() == 1)
        # if single:   # add another dimension for the single env case
        #     rew = rew.unsqueeze(1)
        #     val = val.unsqueeze(1)
        #     done = done.unsqueeze(1)
        #     v_last = v_last.unsqueeze(0) if v_last.dim() == 1 else v_last

        for t in reversed(range(rollout_size)):
            if t == rollout_size - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value.view(-1)
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t] #one step TD error
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        return advantages, advantages + values