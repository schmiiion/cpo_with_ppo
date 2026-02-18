from collections import defaultdict
import numpy as np
from safe_rl_lab.algo.policy_gradient import PolicyGradient
from safe_rl_lab.utils.lagrange import PIDLagrange
from safe_rl_lab.utils.rollout_buffer import RolloutBuffer, PhasicBuffer
import torch



class PPGLag(PolicyGradient):

    def __init__(self, logger, runner, agent, policy_optimizer, value_optimizer, cost_optimizer,cfg, device):
        super().__init__(logger=logger, runner=runner, agent=agent,optimizer=policy_optimizer, cfg=cfg, device=device)
        self.agent: PPGAgent = agent
        self.value_optimizer = value_optimizer
        self.buffer = None
        self.cost_optimizer = cost_optimizer
        self.lagrange = PIDLagrange(
            pid_kp=cfg.algo.k_p,
            pid_ki=cfg.algo.k_i,
            pid_kd=cfg.algo.k_d,
            pid_d_delay=cfg.algo.d_delay,
            pid_delta_d_ema_alpha=cfg.algo.pid_delta_d_ema_alpha,
            pid_delta_p_ema_alpha=cfg.algo.pid_delta_p_ema_alpha,
            sum_norm=cfg.algo.sum_norm,
            diff_norm=cfg.algo.diff_norm,
            penalty_max=cfg.algo.penalty_max,
            lagrangian_multiplier_init=cfg.algo.lagrangian_multiplier_init,
            cost_limit=cfg.algo.cost_limit)

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

        self.ppg_buffer = PhasicBuffer(self.cfg, self.runner.num_envs, self.runner.obs_shape, self.runner.act_shape, device=self.device)

        self.buffer = RolloutBuffer(
            num_steps=self.cfg.algo.rollout_size,
            num_envs=self.runner.num_envs,
            obs_shape=self.runner.obs_shape,
            act_shape=self.runner.act_shape,
            device=self.device,
            use_cost=self.cfg.algo.use_cost,
        )

        for phase in range(num_phases):

            for policy_phase_iteration in range(1, self.cfg.algo.N_pi + 1):
                self.policy_phase(policy_phase_iteration)

            self.auxiliary_phase()



    def policy_phase(self, iteration):
        """
        1. Do rollout
        2. Compute GAE
        3. Do policy optimizatino E_pi times
        4. Do value optimizatino E_v times
        5. Add all (s_t, V_targ) to buffer_B
        """

        if iteration % self.cfg.sps_logging_interval == 0:
            super()._log_sps(self.global_step)

        # 1. Sample Buffer and return bootstrap values
        global_step, rollout_info, last_val, last_done, last_cval = self.runner.run(self.agent, self.buffer,
                                                                                    self.global_step)
        self.global_step = global_step
        self._process_episodic_stats(rollout_info) # self._safe_if_best(global_step)

        #  2. Compute GAE (inside Buffer)
        self.buffer.compute_gae(last_val, last_done, self.cfg.algo.gae.gamma, self.cfg.algo.gae.lam, last_cval)

        data = self.buffer.get()
        # Update Lambdas and compute adv_total
        adv_total = self.update_lambda_and_augment_adv(rollout_info, data)
        data["adv"] = adv_total

        # 3. Policy update loop
        for policy_epoch in range(1, self.cfg.algo.E_pi + 1):
            update_state = self.update_policy(data, rollout_info)
            self.logger.log(metrics=update_state, step=self.global_step, prefix="policy_phase")

        # 4. Value update loop
        for value_epoch in range(1, self.cfg.algo.E_v + 1):
            stats = self.update_value_functions(data)
            self.logger.log(metrics=stats, step=self.global_step, prefix="policy_phase")

        # 5. Populate PPG Buffer
        self.buffer.populate_phasic_buffer(self.ppg_buffer)


    def auxiliary_phase(self):
        # 1. compute pi_theta_old for all states in phasic buffer
        self.ppg_buffer.compute_densities_for_all_states(agent=self.agent, batch_size=self.cfg.algo.N_pi*self.cfg.algo.rollout_size)

        # 2. Distill Features in Policy Network and optimize Value Network
        data = self.ppg_buffer.get()
        dataset_size = data["obs"].shape[0]
        aux_batch_size = dataset_size // (self.cfg.algo.N_pi * self.cfg.algo.aux_mb_per_N_pi)

        phase_stats = defaultdict(list)
        aux_opt_step = 0

        for aux_epoch in range(1, self.cfg.algo.E_aux + 1):
            inds = torch.randperm(dataset_size)

            for start in range(0, dataset_size, aux_batch_size):
                # MB SETUP
                end = start + aux_batch_size
                mb_inds = inds[start:end]

                # RETRIEVE DATA
                mb_obs = data["obs"][mb_inds]
                mb_vtarg = data["v_targ"][mb_inds]
                mb_ctarg = data["c_targ"][mb_inds]
                mb_old_mean = data["old_mean"][mb_inds]
                mb_old_std = data["old_std"][mb_inds]

                # OPTIMIZATION STEPS
                dist_old = torch.distributions.Normal(mb_old_mean, mb_old_std)

                # Update Policy and distill features
                self.optimizer.zero_grad()
                L_joint, joint_metrics = self.compute_L_joint(mb_obs, mb_vtarg, mb_ctarg, dist_old)
                L_joint.backward()
                self.optimizer.step()

                #Update Value Network
                self.value_optimizer.zero_grad()
                L_value = self.compute_L_value_unclipped(mb_obs, mb_vtarg)
                L_value.backward()
                self.value_optimizer.step()
                self.logger.log(metrics={"L_value": L_value.item()}, step=self.global_step, prefix="auxiliary_phase_Critic")

                #Update Cost Network
                self.cost_optimizer.zero_grad()
                L_cost = self.compute_L_value_unclipped(mb_obs, mb_ctarg)
                L_cost.backward()
                self.cost_optimizer.step()
                self.logger.log(metrics={"L_cost": L_cost.item()}, step=self.global_step, prefix="auxiliary_phase_Critic")

                #collect metrics over batches
                phase_stats["L_value"].append(L_value.item())
                phase_stats["L_cost"].append(L_cost.item())
                phase_stats["L_joint"].append(L_joint.item())

                #debug into aux phase:
                self.logger.log(
                    metrics={
                        "debug_aux/ActorValueHeadL": joint_metrics["L_aux_value"],
                        "debug_aux/ActorCostHeadL": joint_metrics["L_aux_cost"],
                        "debug_aux/KL_Divergence": joint_metrics["KL_Div"],
                        "debug_aux/L_value": L_value.item(),
                        "debug_aux/L_cost": L_cost.item(),
                        "debug_aux/L_joint": L_joint.item(),
                        "debug_aux/aux_step": aux_opt_step
                    },
                step=None)
                aux_opt_step += 1

        # --- END OF PHASE: Log Averages to Main Dashboard ---
        avg_stats = {k: np.mean(v) for k, v in phase_stats.items()}
        self.logger.log(metrics=avg_stats, step=self.global_step, prefix="auxiliary_phase")


######################### POLICY RELATED ##################################################

    def update_policy(self, data, rollout_info):
        """
        1. Normalizes the reward
        2. Splits the batch along algo.mini_epochs
        3. Loop over mini epochs
            - Call L^Clip computation
            - Optimize actor with L^clip
        """
        #Create minibatches for |Derived from A.2 Hyperparams: #mb per eoch
        dataset_size = data["obs"].shape[0]
        batch_size = dataset_size // self.cfg.algo.number_mb_per_epoch
        b_inds = np.arange(dataset_size)
        np.random.shuffle(b_inds)
        update_stats = defaultdict(list)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_inds = b_inds[start:end]
            mb = {k: v[mb_inds] for k, v in data.items()}

            loss, stats = self.compute_L_clip(mb)

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.algo.max_grad_norm is not None and self.cfg.algo.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), self.cfg.algo.max_grad_norm)
            self.optimizer.step()

            #collect statsdid
            for k, v in stats.items():
                update_stats[k].append(v)

            if self.cfg.algo.use_kl_early_stopping:
                if stats["approx_kl"] > self.cfg.algo.early_stopping_target_kl:
                    print("early stopping triggered")
                    break
                else:
                    print("continue")

        # aggregate stats after all epochs
        avg_stats = {k: np.mean(v) for k, v in update_stats.items()}
        return avg_stats

    def compute_L_clip(self, batch):
        obs = batch['obs']
        act = batch['act']
        old_logp = batch['logp']
        adv = batch['adv']
        stats = {}

        new_logp, entropy, _ = self.agent.evaluate_actions(obs, act, need_val=False)
        entropy_scalar = entropy.mean()
        stats['entropy'] = entropy_scalar.item()

        logratio = new_logp - old_logp
        ratio = logratio.exp()

        with torch.no_grad():
            approx_kl =((ratio - 1) - logratio).mean().item()
            stats['approx_kl'] = approx_kl

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.cfg.algo.clip_epsilon, 1.0 + self.cfg.algo.clip_epsilon) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        return policy_loss, stats


    def compute_L_joint(self, obs, v_targ, c_targ, dist_old):
        joint_metrics = {}
        #1. Compute L_aux
        policy_val = self.agent.get_policy_value(obs).flatten()
        L_aux_value = 0.5 * ((policy_val - v_targ)**2).mean()
        joint_metrics["L_aux_value"] = L_aux_value.item()

        policy_cost = self.agent.get_policy_cost(obs).flatten()
        L_aux_cost = 0.5 * ((policy_cost - c_targ) ** 2).mean()
        joint_metrics["L_aux_cost"] = L_aux_cost.item()

        dist_new, _  = self.agent.model(obs)
        kl_divergence = torch.distributions.kl_divergence(dist_old, dist_new).mean()
        joint_metrics["KL_Div"] = kl_divergence.item()

        L_joint = L_aux_value + L_aux_cost + self.cfg.algo.beta_clone * kl_divergence

        return L_joint, joint_metrics


######################### VALUE RELATED ##################################################

    def update_value_functions(self, data):
        dataset_size = data["obs"].shape[0]
        batch_size = dataset_size // self.cfg.algo.number_mb_per_epoch
        b_inds = np.arange(dataset_size)
        np.random.shuffle(b_inds)
        update_stats = defaultdict(list)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_inds = b_inds[start:end]
            mb = {k: v[mb_inds] for k, v in data.items()}

            obs = mb['obs']
            v_targ = mb["ret"]
            c_targ = mb["c_ret"]

            L_value = self.compute_L_value_unclipped(obs, v_targ)
            self.value_optimizer.zero_grad()
            L_value.backward()
            if self.cfg.algo.max_grad_norm is not None and self.cfg.algo.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.agent.value_critic.parameters(), self.cfg.algo.max_grad_norm)
            self.value_optimizer.step()
            update_stats["L_value"].append(L_value.item())

            L_cost = self.compute_L_value_unclipped(obs, c_targ)
            self.cost_optimizer.zero_grad()
            L_cost.backward()
            if self.cfg.algo.max_grad_norm is not None and self.cfg.algo.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.agent.cost_critic.parameters(), self.cfg.algo.max_grad_norm)
            self.cost_optimizer.step()
            update_stats["L_cost"].append(L_cost.item())

        # aggregate stats after all epochs
        avg_stats = {k: np.mean(v) for k, v in update_stats.items()}
        return avg_stats


    def compute_L_value_clipped(self, batch):
        obs = batch['obs']
        old_val = batch["val"]
        ret = batch['ret']

        new_val = self.agent.get_value(obs)

        v_loss_unclipped = (new_val - ret) ** 2
        v_clipped = old_val + torch.clamp(
            new_val - old_val, -self.cfg.algo.clip_epsilon, self.cfg.algo.clip_epsilon,
        )
        v_loss_clipped = (v_clipped - ret) ** 2
        v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        return v_loss, {"v_loss": v_loss.item()}


    def compute_L_value_unclipped(self, obs, target, is_cost=False):
        if is_cost:
            val = self.agent.get_cost_value(obs)
        else:
            val = self.agent.get_value(obs)
        L_value = 0.5 * ((val -target)**2).mean()
        return L_value


######################### COST RELATED ##################################################

    def update_value_function123(self, data):
        dataset_size = data["obs"].shape[0]
        batch_size = dataset_size // self.cfg.algo.number_mb_per_epoch
        b_inds = np.arange(dataset_size)
        np.random.shuffle(b_inds)
        update_stats = defaultdict(list)

        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            mb_inds = b_inds[start:end]
            mb = {k: v[mb_inds] for k, v in data.items()}

            loss, stats = self.compute_L_value_clipped(mb)

            self.value_optimizer.zero_grad()
            loss.backward()
            if self.cfg.algo.max_grad_norm is not None and self.cfg.algo.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.agent.value_critic.parameters(), self.cfg.algo.max_grad_norm)
            self.value_optimizer.step()

            #collect statsdid
            for k, v in stats.items():
                update_stats[k].append(v)

        # aggregate stats after all epochs
        avg_stats = {k: np.mean(v) for k, v in update_stats.items()}
        return avg_stats



    def update_lambda_and_augment_adv(self, rollout_info, data):
        # --- SECTION 1: Lagrangian Update ---
        if "scaled_cost" in rollout_info:
            Jc = rollout_info["scaled_cost"]
        else:
            Jc = data["cost"].mean() * self.cfg.env.max_episode_steps

        self.lagrange.update(Jc)
        cur_lambda = self.lagrange.lagrangian_multiplier

        self.logger.log({
            "cost/lambda": cur_lambda,
            "cost/Jc": Jc,
            "cost/cost_violation": Jc - self.cfg.algo.cost_limit
        }, step=self.global_step)

        c_adv = data['c_adv']
        adv = data["adv"]

        if self.cfg.algo.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            c_adv = (c_adv - c_adv.mean())

        adv_total = (adv - cur_lambda * c_adv) / (1 + cur_lambda)
        return adv_total

###### TO MAKE ABC HAPPY:

    def compute_loss(self, batch):
        pass
