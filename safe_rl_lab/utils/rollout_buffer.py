import torch
from typing import Tuple

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape: Tuple, act_shape:Tuple, device="cpu", use_cost=False, use_phasic=False):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        self.ptr = 0
        self.full = False

        #Always needed:
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape, device=device)
        self.act = torch.zeros((num_steps, num_envs) + act_shape, device=device)
        self.rew = torch.zeros((num_steps, num_envs), device=device)
        self.val = torch.zeros((num_steps, num_envs), device=device)
        self.logp = torch.zeros((num_steps, num_envs), device=device)
        self.done = torch.zeros((num_steps, num_envs), device=device)
        self.adv = torch.zeros((num_steps, num_envs), device=device)
        self.ret = torch.zeros((num_steps, num_envs), device=device)

        #for constrained RL:
        self.use_cost = use_cost
        if use_cost:
            self.cost = torch.zeros((num_steps, num_envs), device=device)
            self.cval = torch.zeros((num_steps, num_envs), device=device)
            self.cadv = torch.zeros((num_steps, num_envs), device=device)
            self.cret = torch.zeros((num_steps, num_envs), device=device)

        #for phasic updates (PPG)
        self.use_phasic = use_phasic
        if use_phasic:
            self.pd_mean = torch.zeros((num_steps, num_envs) + act_shape, device=device)
            self.pd_std = torch.zeros((num_steps, num_envs) + act_shape, device=device)

    def store(self, obs, act, rew, val, logp, done, cost=None, cval=None, pd_mean=None, pd_std=None):
        """Save one step of samples from the env"""
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.val[self.ptr] = val
        self.logp[self.ptr] = logp
        self.done[self.ptr] = done

        if self.use_cost:
            self.cost[self.ptr] = cost
            self.cval[self.ptr] = cval

        if self.use_phasic:
            self.pd_mean[self.ptr] = pd_mean
            self.pd_std[self.ptr] = pd_std

        self.ptr += 1


    def compute_gae(self, last_val, last_done, gamma, gae_lambda, last_cval=None):
        """Compute GAE in-place for rewards and optionally for costs"""
        #1. Reward GAE
        self._compute_gae_internal(
            self.rew, self.val, last_val, self.done, last_done, self.adv, self.ret, gamma, gae_lambda
        )

        #2. Cost GAE (if needed)
        if self.use_cost:
            self._compute_gae_internal(
                self.cost, self.cval, last_cval, self.done, last_done, self.cadv, self.cret, gamma, gae_lambda
            )


    def _compute_gae_internal(self, rews, vals, last_val, dones, last_done, advs, rets, gamma, gae_lambda):
        """In-place computation. Awesome!!!"""
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_non_terminal = 1.0 - last_done
                next_val = last_val
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_val = vals[t + 1]

            delta = rews[t] + gamma * next_val * next_non_terminal - vals[t]
            advs[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

        #compute Returns = Advantage + Value
        rets[:] = advs + vals


    def get(self):
        """Return flattened data (more iid-ish) for training"""
        self.ptr = 0
        data = {
            "obs": self.obs.flatten(0,1),
            "act": self.act.flatten(0,1),
            "ret": self.ret.flatten(0,1),
            "adv": self.adv.flatten(0, 1),
            "logp": self.logp.flatten(0, 1),
            "val": self.val.flatten(0, 1),
        }

        if self.use_cost:
            data["cost_ret"] = self.cret.flatten(0, 1)
            data["cost_adv"] = self.cadv.flatten(0, 1)
            data["cval"] = self.cval.flatten(0, 1)

        if self.use_phasic:
            data["pd_mean"] = self.pd_mean.flatten(0, 1)
            data["pd_std"] = self.pd_std.flatten(0, 1)

        return data

#OLD GAE that worked
    #
    # def gae_from_rollout(buf, rollout_size, gamma, gae_lambda):
    #     """:returns advantages, returns with same shape as buffer["reward"] / ["val"]"""
    #     with torch.no_grad():
    #         rewards = buf["rew"]
    #         vpreds = buf["vpred"]
    #         dones = buf["done"]
    #
    #         # Reward bootstrapping
    #         next_value = buf["v_last"]
    #         next_done = buf["done_last"]
    #
    #         # Cost bootstrapping
    #         costs = buf["cost"]  # Raw signal
    #         cpreds = buf["cpred"]  # Critic predictions
    #         next_cpred = buf["cpred_last"]  # Bootstrapped value
    #
    #         # Output buffers
    #         adv_r = torch.zeros_like(rewards)
    #         adv_c = torch.zeros_like(costs)
    #
    #         lastgaelam_r = 0
    #         lastgaelam_c = 0
    #
    #         for t in reversed(range(rollout_size)):
    #             if t == rollout_size - 1:
    #                 nextnonterminal = 1.0 - next_done
    #                 next_val_r = next_value
    #                 next_val_c = next_cpred
    #             else:
    #                 nextnonterminal = 1.0 - dones[t + 1]
    #                 next_val_r = vpreds[t + 1]
    #
    #                 next_val_c = cpreds[t + 1]
    #
    #             # Reward GAE
    #             delta_r = rewards[t] + gamma * next_val_r * nextnonterminal - vpreds[t]
    #             adv_r[t] = lastgaelam_r = delta_r + gamma * gae_lambda * nextnonterminal * lastgaelam_r
    #
    #             # Cost GAE
    #             delta_c = costs[t] + gamma * next_val_c * nextnonterminal - cpreds[t]
    #             adv_c[t] = lastgaelam_c = delta_c + gamma * gae_lambda * nextnonterminal * lastgaelam_c
    #
    #         returns_r = adv_r + vpreds
    #         returns_c = adv_c + cpreds
    #
    #         return adv_r, returns_r, adv_c, returns_c