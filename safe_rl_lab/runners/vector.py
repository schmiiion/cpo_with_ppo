import torch
import numpy as np


class VectorRunner:

    def __init__(self, envs, model, obs_dims, act_dims, device="cpu"):
        self.envs = envs
        self.model = model
        self.obs_dims = (obs_dims,) if isinstance(obs_dims, int) else tuple(obs_dims)
        self.act_dims = (act_dims,) if isinstance(act_dims, int) else tuple(act_dims)
        self.device = device
        self.obs, _ = self.envs.reset()
        self.obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device)


    @torch.inference_mode()
    def run(self, T):
        self.model.eval()
        N = self.obs.shape[0]

        obs_buf = torch.empty((T, N, *self.obs_dims), dtype=torch.float32, device=self.device)
        act_buf = torch.empty((T, N, *self.act_dims), dtype=torch.float32, device=self.device)
        logprob_buf = torch.empty((T, N), dtype=torch.float32, device=self.device)
        val_buf = torch.empty((T, N), dtype=torch.float32, device=self.device)
        rew_buf = torch.empty((T, N), dtype=torch.float32, device=self.device)
        cost_buf = torch.empty((T, N), dtype=torch.float32, device=self.device)
        done_buf = torch.empty((T, N), dtype=torch.bool, device=self.device)
        tout_buf = torch.empty((T, N), dtype=torch.bool, device=self.device)

        ep_events = []
        for t in range(T):
            obs_buf[t].copy_(self.obs)
            # Policy (batched over envs) -> returns
            action, logprob = self.model.sample_action_and_logp(self.obs)
            act_buf[t] = action
            logprob_buf[t] = logprob
            assert torch.isfinite(logprob).all()
            assert (action.abs() <= 1 + 1e-6).all()
            # Value
            val = self.model.forward_critic(self.obs)
            val_buf[t] = val

            obs_next, reward, terminated, truncated, infos = self.envs.step(action.detach().cpu().numpy())
            obs_next = torch.as_tensor(obs_next, dtype=torch.float32, device=self.device)

            final_infos = infos.get("final_info")
            if final_infos is not None:
                for i, fi in enumerate(final_infos):
                    if fi is not None:
                        ep = fi.get("episode", {})
                        ep_ret = ep.get("r")
                        ep_len = ep.get("l")
                        ep_events.append({
                            "env_idx":i,
                            "return":ep_ret,
                            "length":ep_len,
                        })

            rew_buf[t] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            cost_buf[t] = torch.as_tensor(infos["cost"], dtype=torch.float32, device=self.device)
            done_buf[t] = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
            tout_buf[t] = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)

            self.obs = obs_next

        v_last = self.model.forward_critic(self.obs) # dimension needs to be [N]

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
            "episodes": ep_events,
        }