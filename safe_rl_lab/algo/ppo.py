import torch
from safe_rl_lab.models.actor_critic import ActorCritic
from safe_rl_lab.runners.single import SingleRunner


class PPO:

    def __init__(self, env, model="A2C", runner_type="single", obs_dim=56, act_dim=8,
                 rollout_size=4, total_steps=10000, gamma=0.99, lam=0.8):
        self.model = ActorCritic(obs_dim=obs_dim, act_dim=act_dim) if model == "A2C" else None
        self.runner = SingleRunner(env, self.model, obs_dim, act_dim) if runner_type == "single" else None
        self.rollout_size = rollout_size
        self.total_steps = total_steps
        self.gamma = gamma
        self.lam = lam

    def train(self, env):
        steps_done = 0
        while steps_done < self.total_steps:
            buffer:dict = self.runner.run(self.rollout_size)
            adv, returns = self._gae_from_rollout(buffer)

            #normalize advantages per rollout - WHY?
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            obs = buffer["obs"]
            act = buffer["act"]
            logp0 = buffer["logprob"]   #old values
            val0 = buffer["val"]        #old values

            self._ppo_update()

            steps_done += self.rollout_size

    def _ppo_update(self):
        pass

    @torch.no_grad()
    def _gae_from_rollout(self, buf):
        # buf: {"rew":[T], "val":[T], "done":[T], "v_last": scalar}
        rew = buf["rew"]  # float32 [T]
        val = buf["val"]  # float32 [T]
        done = buf["done"]  # bool    [T] (true terminal only)
        v_last = buf["v_last"]  # float32 []

        T = rew.shape[0]
        adv = torch.zeros_like(rew)
        next_adv = torch.zeros((), device=rew.device) #running accumulator
        next_v = v_last #step init to run backwards
        not_done = (~done).float()

        for t in reversed(range(T)):
            delta = rew[t] + self.gamma * not_done[t] * next_v - val[t]
            next_adv = delta + self.gamma * self.lam * not_done[t] * next_adv
            adv[t] = next_adv
            next_v = val[t]

        ret = adv + val
        return adv, ret
