import torch
import numpy as np

class VectorRunner:

    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device

        #Reset once
        self.obs, _ = self.env.reset()
        self.obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
        self.done = torch.zeros(env.num_envs, device=self.device)

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def obs_shape(self):
        return self.env.single_observation_space.shape

    @property
    def act_shape(self):
        return self.env.single_action_space.shape

    @torch.inference_mode()
    def run(self, agent, buffer, global_step):
        """Drives the agent in the env and fills the buffer"""
        num_steps = buffer.num_steps
        ep_info = []

        for step in range(0, num_steps):
            global_step += self.env.num_envs

            #1 Action selection
            if buffer.use_phasic:
                action, info = agent.act(self.obs, return_dist_params=True)
            else:
                action, info = agent.act(self.obs)

            #2 Environment Step
            next_obs, reward, terminated, truncated, infos = self.env.step(action)

            # COLLECT: Capture episodic info when an episode ends
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # We just pass the whole dict back.
                        # This automatically includes 'episode_cost' if the env provides it.
                        ep_info.append(info)

            cval = info["cval"] if buffer.use_cost else None

            # 3 Store data
            buffer.store(
                obs=self.obs,
                act=action,
                rew=torch.tensor(reward, device=self.device),
                val=info["val"],
                logp=info["logp"],
                done=self.done,
                cost=torch.as_tensor(infos["cost"], device=self.device) if buffer.use_cost else None,
                cval=cval,
                pd_mean=info.get("pd_mean"),
                pd_std=info.get("pd_std"),
            )

            #4 Advance
            self.obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            self.done = torch.as_tensor(np.logical_or(terminated, truncated), dtype=torch.float32, device=self.device)

        ###Bootstrapping
        last_val = agent.get_value(self.obs).flatten()
        last_cval = info["cval"] if buffer.use_cost else None

        return global_step, ep_info, (last_val, self.done, last_cval)
