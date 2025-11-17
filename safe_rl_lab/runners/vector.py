import torch
import numpy as np
import wandb


class VectorRunner:

    def __init__(self, envs, agent, obs_dims, act_dims, device="cpu"):
        self.envs = envs
        self.agent = agent
        self.obs_dims = (obs_dims,) if isinstance(obs_dims, int) else tuple(obs_dims)
        self.act_dims = (act_dims,) if isinstance(act_dims, int) else tuple(act_dims)
        self.device = device
        self.obs, _ = self.envs.reset()
        self.done = torch.zeros(envs.num_envs)
        self.obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
        self.num_envs = envs.num_envs


    @torch.inference_mode()
    def run(self, num_steps, global_step, agent):
        N = self.num_envs

        obs_buf = torch.empty((num_steps, N, *self.obs_dims), dtype=torch.float32, device=self.device)
        act_buf = torch.empty((num_steps, N, *self.act_dims), dtype=torch.float32, device=self.device)
        logprob_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        val_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        rew_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        cost_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        done_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        ep_rewards = []

        for step in range(0, num_steps):
            global_step += self.num_envs
            obs_buf[step] = self.obs
            done_buf[step] = self.done

            # Action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(self.obs)
                val_buf[step] = value.flatten()

            #Monitor actions????
            a = action.detach().cpu().numpy()
            low, high = self.envs.single_action_space.low, self.envs.single_action_space.high

            # count out-of-bounds samples
            # too_low = (a < low).sum()
            # too_high = (a > high).sum()
            # total = np.prod(a.shape)
            # frac_oob = (too_low + too_high) / total
            #print(f"Out-of-bounds actions: {frac_oob * 100:.2f}%")

            act_buf[step] = action
            logprob_buf[step] = logprob

            obs_next, reward, terminations, truncations, infos = self.envs.step(action.detach().cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rew_buf[step] = torch.tensor(reward).view(-1)
            cost_buf[step] = torch.tensor(infos["cost"])
            self.obs, self.done = torch.Tensor(obs_next), torch.Tensor(next_done)

            if "final_info" in infos: # wenn der key da ist
                for info in infos["final_info"]: #drauf zugreifen und iterieren
                    if info and "episode" in info:
                        ep_rewards.append(float(info["episode"]["r"]))
                        if wandb.run:
                            wandb.log({"charts/episodic_return": info["episode"]["r"]}, step=global_step)

        v_last = agent.get_value(self.obs)
        done_last = self.done
        ep_rewards_mean = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0

        if ep_rewards and wandb.run:
            mean_r = float(np.mean(ep_rewards))
            std_r = float(np.std(ep_rewards))
            wandb.log({
                "charts/ep_return_mean": mean_r,
                "charts/ep_return_std": std_r,
            }, step=global_step)

        return {
            "obs": obs_buf,
            "act": act_buf,
            "logprob": logprob_buf,
            "val": val_buf,
            "rew": rew_buf,
            "cost": cost_buf,
            "done": done_buf,
            "v_last": v_last,
            "done_last": done_last,
            "ep_rewards_mean": ep_rewards_mean,
        }, global_step