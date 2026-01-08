import torch

@torch.no_grad()
def gae_from_rollout(buf, rollout_size, gamma, gae_lambda):
    """:returns advantages, returns with same shape as buffer["reward"] / ["val"]"""
    with torch.no_grad():
        rewards = buf["rew"]  # [T] or [T, N]
        values = buf["val"]  # [T] or [T, N]
        dones = buf["done"]  # [T]  or [T, N] -> (true terminal only)
        next_value = buf["v_last"]  # scalar of [N]
        next_cost = buf["cpred_last"]
        next_done = buf["done_last"]
        costs = buf["cost"]
        cpreds = buf["cpred"]
        advantages = torch.zeros_like(rewards)
        advantages_cost = torch.zeros_like(costs)
        lastgaelam = 0


        for t in reversed(range(rollout_size)):
            if t == rollout_size - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
                next_cost = next_cost
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
                next_costs = costs[t + 1]
            delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t] #one step TD error
            delta_cost = costs[t] + gamma * next_costs * nextnonterminal - cpreds[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            advantages_cost[t] = lastgaelam = delta_cost + gamma * gae_lambda * nextnonterminal * lastgaelam
        return advantages, advantages + values, advantages_cost, advantages_cost +costs