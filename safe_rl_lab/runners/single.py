import torch

class SingleRunner:
    
    def __init__(self, env, model, obs_dims, act_dims, device="cpu"):
        self.env = env
        self.model = model
        self.obs_dims = (obs_dims,) if isinstance(obs_dims, int) else tuple(obs_dims)
        self.act_dims = (act_dims,) if isinstance(act_dims, int) else tuple(act_dims)
        self.device = device
        self.obs = None

    def reset(self):
        self.obs, _ = self.env.reset()
        self.obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)
        return

    def run(self, k_steps):
        self.model.eval()
        with torch.inference_mode():
            if self.obs is None:
                self.reset()

            obs_buf = torch.empty((k_steps, *self.obs_dims), dtype=torch.float32, device=self.device)
            act_buf = torch.empty((k_steps, *self.act_dims), dtype=torch.float32, device=self.device)  # adjust for your action space
            logprob_buf = torch.empty((k_steps,), dtype=torch.float32, device=self.device)
            val_buf = torch.empty((k_steps,), dtype=torch.float32, device=self.device)
            rew_buf = torch.empty((k_steps,), dtype=torch.float32, device=self.device)
            cost_buf = torch.empty((k_steps,), dtype=torch.float32, device=self.device)
            done_buf = torch.empty((k_steps,), dtype=torch.bool, device=self.device)
            tout_buf = torch.empty((k_steps,), dtype=torch.bool, device=self.device)

            v_last = None

            for t in range(k_steps):
                obs_buf[t].copy_(self.obs)
                #Policy
                mu, std = self.model.forward_actor(self.obs)
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
                act_buf[t] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
                logprob = dist.log_prob(action).sum(-1)
                logprob_buf[t] = logprob
                #Value
                val = self.model.forward_critic(self.obs)
                val_buf[t] = val

                a_cpu = action.detach().cpu().numpy()
                obs_next, reward, terminated, truncated, info = self.env.step(a_cpu)
                obs_next = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)

                rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
                cost_buf[t] = torch.as_tensor(info["cost"], dtype=torch.float32, device=self.device)
                done_buf[t] = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
                tout_buf[t] = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

                #value estimate for next state s_t+1 - needed for bootstrap if rollout ends
                v_next = self.model.forward_critic(obs_next).squeeze(-1)
                v_last = torch.zeros_like(v_next) if terminated else v_next

                if terminated or truncated:
                    self.reset()
                else:
                    self.obs = obs_next

            return {
                "obs": obs_buf,
                "act": act_buf,
                "logprob": logprob_buf,
                "val": val_buf,
                "rew": rew_buf,
                "cost": cost_buf,
                "done": done_buf,
                "tout": tout_buf,
                "v_last": v_last,
            }