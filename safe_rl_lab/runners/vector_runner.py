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
    def run(self, num_steps, global_step, agent, is_phasic=False):
        N = self.num_envs

        obs_buf = torch.empty((num_steps, N, *self.obs_dims), dtype=torch.float32, device=self.device)
        act_buf = torch.empty((num_steps, N, *self.act_dims), dtype=torch.float32, device=self.device)
        logprob_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        val_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        rew_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        cost_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        done_buf = torch.empty((num_steps, N), dtype=torch.float32, device=self.device)
        if is_phasic:
            pd_mean_buf = torch.empty((num_steps, N, *self.act_dims), dtype=torch.float32, device=self.device)
            pd_std_buf = torch.empty((num_steps, N, *self.act_dims), dtype=torch.float32, device=self.device)
        ep_rewards, ep_lengths, ep_costs = [], [], []

        for step in range(0, num_steps):
            global_step += self.num_envs
            obs_buf[step] = self.obs
            done_buf[step] = self.done

            # Action logic
            with torch.no_grad():
                if is_phasic:
                    action, info_dict = agent.act(self.obs, return_dist_params=True)
                    pd_mean_buf[step] = info_dict["pd_mean"]
                    pd_std_buf[step] = info_dict["pd_std"]
                else:
                    action, info_dict = agent.act(self.obs)
                value = info_dict["vpred"]
                val_buf[step] = value.flatten()
                logprob = info_dict["logp"]

            act_buf[step] = action
            logprob_buf[step] = logprob

            obs_next, reward, terminations, truncations, infos = self.envs.step(action.detach().cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rew_buf[step] = torch.tensor(reward).view(-1)
            cost_buf[step] = torch.as_tensor(infos["cost"], dtype=torch.float32, device=self.device)
            self.obs, self.done = torch.Tensor(obs_next), torch.Tensor(next_done)

            if "final_info" in infos: # wenn der key da ist
                for info in infos["final_info"]: #drauf zugreifen und iterieren
                    if info and "episode" in info:
                        ep_rewards.append(float(info["episode"]["r"]))
                        ep_lengths.append(float(info["episode"]["l"]))
                        ep_costs.append(float(info["final_cost_sum"]))
                        if wandb.run:
                            wandb.log({
                                "charts/episodic_return": info["episode"]["r"],
                                "charts/episodic_length": info["episode"]["l"],
                                "charts/episodic_cost": info["final_cost_sum"],
                            }, step=global_step)

        v_last = agent.v(self.obs)
        done_last = self.done
        ep_rewards_mean = sum(ep_rewards) / len(ep_rewards) if ep_rewards else 0.0

        if ep_rewards and wandb.run:
            wandb.log({
                "charts/ep_return_mean": float(np.mean(ep_rewards)),
                "charts/ep_length_mean": float(np.mean(ep_lengths)),
                "charts/ep_return_std": float(np.std(ep_rewards)),
                "charts/ep_cost_mean": float(np.mean(ep_costs)),
            }, step=global_step)

        return_dict = {
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
        }
        if is_phasic:
            return_dict["old_pd_mean"] = pd_mean_buf
            return_dict["old_pd_std"] = pd_std_buf

        return return_dict, global_step