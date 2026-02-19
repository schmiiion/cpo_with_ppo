import torch
import numpy as np

class VectorRunner:

    def __init__(self, env, device="cpu", cfg=None):
        self.env = env
        self.device = device
        self.cfg = cfg

        #Reset once
        self.obs, _ = self.env.reset(seed=cfg.seed)
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
        rollout_info = {}

        for step in range(0, num_steps):
            global_step += self.env.num_envs
            action, agent_info = agent.step(self.obs)

            #2 Environment Step
            cpu_action = action.cpu().numpy()
            next_obs, reward, terminated, truncated, info = self.env.step(cpu_action)

            done_bool = np.logical_or(terminated, truncated)
            current_done = torch.as_tensor(done_bool, dtype=torch.float32, device=self.device)

            cval, cost_tensor = None, None
            if buffer.use_cost:
                costs = info["cost"] * self.cfg.algo.cost_scaling
                cost_tensor = torch.as_tensor(costs, device=self.device)
                if self.cfg.algo.a2c_architecture == "separate":
                    cval = agent.get_cost_value(self.obs)
                else:
                    cval = agent_info["cval"]


            # 4. Collection: Capture episodic info
            if "final_info" in info:
                rollout_info = {}
                acc_cost, acc_rew = 0, 0
                for vec_env in info["final_info"]:
                    acc_cost += vec_env["acc_cost"]
                    acc_rew += vec_env["episode"]["r"]
                num_finished = len(info["final_info"])
                rollout_info["rew"] = (acc_rew / num_finished)
                raw_cost = (acc_cost / num_finished)
                rollout_info["raw_cost"] = raw_cost
                rollout_info["scaled_cost"] = raw_cost * self.cfg.algo.cost_scaling

            # 3 Store data
            buffer.store(
                obs=self.obs,
                act=action,
                rew=torch.tensor(reward, device=self.device),
                val=agent_info["val"],
                logp=agent_info["logp"],
                done=current_done,
                cost=cost_tensor,
                cval=cval,
            )

            #4 Advance
            self.obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
            self.done = current_done

        ###Bootstrapping
        last_val = agent.get_value(self.obs).flatten()
        last_cval = agent.get_cost_value(self.obs).flatten() if buffer.use_cost else None

        #average stats
        # avg_rollout_info: dict = VectorRunner.flatten_rollout_stats(rollout_info)

        return global_step, rollout_info, last_val, self.done, last_cval

    @staticmethod
    def flatten_rollout_stats(infos) -> dict:
        if not infos["final_info"]:
            return {}

        keys = infos[0].keys()
        avg_stats = {
            f"{k}_mean": np.mean([d[k] for d in infos])
            for k in keys
        }
        return avg_stats