from safe_rl_lab.algo.policy_gradient import PolicyGradient
from safe_rl_lab.utils.rollout_buffer import RolloutBuffer, PhasicBuffer
import torch



class PPG(PolicyGradient):

    def __init__(self, logger, runner, agent, optimizer, cfg, device):
        super().__init__(runner, logger, cfg, device)
        self.agent = agent
        self.optimizer = optimizer
        self.buffer = None


    def learn(self):
        """
        Train loop for ...in phase/ update:
            I.init buffer_B
            II.for ... in 1 - N_pi (policy phase):
                1. Perform rollouts
                2. Compute GAE
                3. for ... in E_pi (policy epochs):
                    -OPtimize actor with L^clip
                4. for ... in E_v (value epochs):
                    -Optimize critic with L^value
                5. Add all (s_t, V_targ to buffer_B)
            III. Compute and store pi_old pds for pi(.|s) for all s in buffer_B
            IV. for ... in 1 - E_aux:
                1. Optimize L^joint
                2. Optimize L^Value
        """
        num_phases = self.cfg.total_steps // (self.cfg.algo.rollout_size * self.cfg.algo.N_pi)

        self.ppg_buffer = PhasicBuffer(self.cfg, device=self.device)

        self.buffer = RolloutBuffer(
            num_steps=self.cfg.algo.rollout_size,
            num_envs=self.runner.num_envs,
            obs_shape=self.runner.obs_shape,
            act_shape=self.runner.act_shape,
            device=self.device,
            use_cost=self.cfg.algo.use_cost,
            use_phasic=self.cfg.algo.use_phasic,
        )

        for phase in range(num_phases):

            for policy_phase_iteration in range(1, self.cfg.algo.N_pi + 1):

                self.policy_phase(policy_phase_iteration)


            # UPDATE - Template Method
            buffer_data = self.buffer.get()
            update_state = self.update(buffer_data, rollout_info)
            self.logger.log(metrics=update_state, step=self.global_step, prefix="train")

    def policy_phase(self, iteration):

        if iteration % self.cfg.sps_logging_interval == 0:
            super()._log_sps(self.global_step)

        # 1. Sample Buffer and return bootstrap values
        global_step, rollout_info, last_val, last_done, last_cval = self.runner.run(self.agent, self.buffer,
                                                                                    self.global_step)
        self.global_step = global_step
        # log stats and safe a new best modelx
        self._process_episodic_stats(rollout_info)
        # self._safe_if_best(global_step)

        #  2. Compute GAE (inside Buffer)
        self.buffer.compute_gae(last_val, last_done, self.cfg.algo.gae.gamma, self.cfg.algo.gae.lam, last_cval)

        for policy_epoch in range(1, self.cfg.algo.E_pi + 1):
            self.update_policy()

    def auxiliary_phase(self):
        pass

    def update_policy(self, data, rollout_info):
        """Implements L^clip from Cobbe et al."""
        obs = data['obs']
        act = data['act']
        old_logp = data['logp']
        adv = data['adv']

        new_logp, entropy, _ = self.agent.evaluate(obs, act)
        entropy_scalar = entropy.mean()

        logratio = new_logp - old_logp
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl =((ratio - 1) - logratio).mean().item()

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.algo.clip_eps, 1.0 + self.cfg.algo.clip_eps)
        policy_loss = -torch.min(surr1, surr2).mean()

        self.agent.


    def update_critic(self):
        pass
