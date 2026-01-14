from abc import ABC, abstractmethod
import torch
from safe_rl_lab.algo.base_algo import BaseAlgo
from safe_rl_lab.utils.rollout_buffer import RolloutBuffer

class PolicyGradient(BaseAlgo):
    """
    agent is a container to hold the policy and value function. Either shared or disjoint.
    """
    def __init__(self, runner, agent, optimizer, logger, cfg, device="cpu"):
        super().__init__(runner, logger, cfg, device)
        self.agent = agent
        self.optimizer = optimizer
        self.buffer = None

    def learn(self):
        """
        Main training loop:
        1. Sample buffer from environment
        2. Compute GAE
        3. Compute policy gradient (Loss?)
        4. Update policy
        """
        num_updates = self.cfg.total_steps // self.cfg.rollout_size
        global_step = 0

        self.buffer = RolloutBuffer(
            num_steps=self.cfg.rollout_size,
            num_envs=self.runner.num_envs,
            obs_shape=self.runner.obs_shape,
            act_shape=self.runner.act_shape,
            device=self.device,
            use_cost=self.cfg.use_cost,
            use_phasic=self.cfg.use_phasic,
        )

        for update in range(1, num_updates + 1):

            if update % self.cfg.sps_logging_interval == 0:
                super()._log_sps(global_step)

            self._anneal_lr()

            global_step, ep_info, bootstrap_tup = self.runner.run(self.agent, self.buffer, global_step)
            #storing model based on ep_info is greedy, could just be one lucky run that ended. -> from deque???

            last_val, last_done, last_cval = bootstrap_tup
            self.buffer.compute_gae(last_val, last_done, self.cfg.gamma, self.cfg.gae_lambda, last_cval)


    def get_action(self, obs):
        """Interface for the Runner to interact with the Agent."""
        with torch.no_grad():
            if isinstance(obs, torch.Tensor):
                obs = obs.to(self.device)
            else:
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

            return self.agent.get_action(obs)

    def _anneal_lr(self):
        pass
