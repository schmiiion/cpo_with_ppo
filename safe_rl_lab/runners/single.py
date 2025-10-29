import torch

class singleRunner():
    
    def __init__(self, env, device="cpu"):
        self.env = env
        self.device = device

    def run(self, k_steps):
        #store tuples (s_t, a_t,
        obs, info = self.env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        terminated = truncated = False # terminated is success/failure do to the self.env.; Truncated -> stopped for external reasons

        obs_buf = torch.empty((k_steps, *obs.shape), dtype=torch.float32, device=self.device)
        act_buf = torch.empty((k_steps,), dtype=torch.float32, device=self.device)  # adjust for your action space
        rew_buf = torch.empty((k_steps,), dtype=torch.float32, device=self.device)
        cost_buf = torch.empty((k_steps,), dtype=torch.float32, device=self.device)

        for t in range(k_steps):
            obs_buf[t].copy_(obs)
            act = self.env.action_space.sample() #later by model
            obs_next, reward, terminated, truncated, info = self.env.step(act)

            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            cost_buf[t] = torch.as_tensor(info["cost"], dtype=torch.float32, device=self.device)
            act_buf[t] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
            obs = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)

            if terminated or truncated:
                print(f"Return: {ep_return.item():.2f}, Cost: {ep_cost.item():.2f}")
                observation, info = self.env.reset()
                return observation, info, ep_return, ep_cost
        self.env.close()