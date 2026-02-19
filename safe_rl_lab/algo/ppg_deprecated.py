# from collections import defaultdict
#
# import numpy as np
# from torch.utils.data import DataLoader, TensorDataset
#
# from safe_rl_lab.algo.policy_gradient import PolicyGradient
# from safe_rl_lab.utils.rollout_buffer import RolloutBuffer, PhasicBuffer
# import torch
#
#
#
# class PPG(PolicyGradient):
#
#     def __init__(self, logger, runner, a2c, cfg, device):
#         super().__init__(logger=logger, runner=runner, a2c=a2c, cfg=cfg, device=device)
#         self.buffer = None
#         self.num_phases = self.cfg.total_steps // (self.cfg.algo.rollout_size * self.cfg.algo.N_pi)
#         self.ppg_buffer = PhasicBuffer(self.cfg, self.runner.num_envs, self.runner.obs_shape, self.runner.act_shape,
#                                        device=self.device)
#
#
#     def learn(self):
#         """
#         Train loop for ...in phase/ update:
#             I.init buffer_B
#             II.for ... in 1 - N_pi (policy phase):
#                 1. Perform rollouts
#                 2. Compute GAE
#                 3. for ... in E_pi (policy epochs):
#                     -OPtimize actor with L^clip
#                 4. for ... in E_v (value epochs):
#                     -Optimize critic with L^value
#                 5. Add all (s_t, V_targ to buffer_B)
#             III. Compute and store pi_old pds for pi(.|s) for all s in buffer_B
#             IV. for ... in 1 - E_aux:
#                 1. Optimize L^joint
#                 2. Optimize L^Value
#         """
#
#
#         self.buffer = RolloutBuffer(
#             num_steps=self.cfg.algo.rollout_size,
#             num_envs=self.runner.num_envs,
#             obs_shape=self.runner.obs_shape,
#             act_shape=self.runner.act_shape,
#             device=self.device,
#             use_cost=self.cfg.algo.use_cost,
#         )
#
#         for phase in range(self.num_phases):
#
#             for policy_phase_iteration in range(1, self.cfg.algo.N_pi + 1):
#
#                 self.policy_phase(policy_phase_iteration)
#
#             for aux_phase_iteration in range(1, self.cfg.algo.E_aux + 1):
#
#                 self.auxiliary_phase(aux_phase_iteration)
#
#
#
#     def policy_phase(self, iteration):
#         """
#         1. Do rollout
#         2. Compute GAE
#         3. Do policy optimizatino E_pi times
#         4. Do value optimizatino E_v times
#         5. Add all (s_t, V_targ) to buffer_B
#         """
#
#         if iteration % self.cfg.sps_logging_interval == 0:
#             super()._log_sps(self.global_step)
#
#         # 1. Sample Buffer and return bootstrap values
#         global_step, rollout_info, last_val, last_done, last_cval = self.runner.run(self._actor_critic, self.buffer, self.global_step)
#         self.global_step = global_step
#         self._process_episodic_stats(rollout_info) # self._safe_if_best(global_step)
#
#         #  2. Compute GAE (inside Buffer)
#         self.buffer.compute_gae(last_val, last_done, self.cfg.algo.gae.gamma, self.cfg.algo.gae.lam, last_cval)
#
#         # 3. Policy update loop
#         for policy_epoch in range(1, self.cfg.algo.E_pi + 1):
#             stats = self.update_policy()
#             self.logger.log(metrics=stats, step=self.global_step, prefix="policy_phase")
#
#         # 4. Value update loop
#         for value_epoch in range(1, self.cfg.algo.E_v + 1):
#             stats = self.update_value_functions()
#             self.logger.log(metrics=stats, step=self.global_step, prefix="policy_phase")
#
#         # 5. Populate PPG Buffer
#         self.buffer.populate_phasic_buffer(self.ppg_buffer)
#
#
#     def auxiliary_phase(self, iteration):
#         # 1. compute pi_theta_old for all states in phasic buffer
#         self.ppg_buffer.compute_densities_for_all_states(agent=self._actor_critic, batch_size=self.cfg.algo.N_pi*self.cfg.algo.rollout_size)
#
#         # 2. Distill Features in Policy Network and optimize Value Network
#         data = self.ppg_buffer.get()
#         dataset_size = data["obs"].shape[0]
#         aux_batch_size = dataset_size // (self.cfg.algo.N_pi * self.cfg.algo.aux_mb_per_N_pi)
#
#         for aux_epoch in range(1, self.cfg.algo.E_aux + 1):
#             mb_inds = torch.randperm(dataset_size)
#
#             for start in range(0, dataset_size, aux_batch_size):
#                 end = start + aux_batch_size
#                 mb_inds = mb_inds[start:end]
#
#                 mb_obs = data["obs"][mb_inds]
#                 mb_vtarg = data["v_targ"][mb_inds]
#                 mb_old_mean = data["old_mean"][mb_inds]
#                 mb_old_std = data["old_std"][mb_inds]
#
#                 #Distill features through policy head in Policy Network
#                 dist_old = torch.distributions.Normal(mb_old_mean, mb_old_std)
#
#                 self.optimizer.zero_grad()
#                 L_joint = self.compute_L_joint(mb_obs, mb_vtarg, dist_old)
#                 L_joint.backward()
#                 self.optimizer.step()
#                 self.logger.log(metrics={"L_joint":L_joint.item()}, step=self.global_step, prefix="auxiliary_phase" )
#
#                 #Update Value Network
#                 self.value_optimizer.zero_grad()
#                 L_value = self.compute_L_value_unclipped(mb_obs, mb_vtarg)
#                 L_value.backward()
#                 self.value_optimizer.step()
#                 self.logger.log(metrics={"L_value": L_value.item()}, step=self.global_step, prefix="auxiliary_phase")
#
#
#
# ######################### POLICY RELATED ##################################################
#
#     def update_policy(self):
#         """
#         1. Normalizes the reward
#         2. Splits the batch along algo.mini_epochs
#         3. Loop over mini epochs
#             - Call L^Clip computation
#             - Optimize actor with L^clip
#         """
#         update_stats = defaultdict(list)
#
#         data = self.buffer.get()
#         obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
#             data['obs'],
#             data['act'],
#             data['logp'],
#             data['target_value_r'],
#             data['target_value_c'],
#             data['adv_r'],
#             data['adv_c'],
#         )
#
#         dataloader = DataLoader(
#             dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
#             batch_size=self.cfg.algo.batch_size,
#             shuffle=True,
#         )
#
#
#         for (obs, act, logp, target_value_r, target_value_c, adv_r, adv_c,) in dataloader:
#
#             if self.cfg.algo.a2c_architecture == "separate":
#                 pi_loss_item, loss_update_stats = self._update_actor(obs, act, logp, adv_r, adv_c)
#                 update_stats["pi_loss"].append(pi_loss_item)
#                 for key, value in loss_update_stats.items():
#                     update_stats[key].append(value)
#
#         avg_stats = {k: np.mean(v) for k, v in update_stats.items()}
#         return avg_stats
#
#     def compute_L_clip(self, batch):
#         obs = batch['obs']
#         act = batch['act']
#         old_logp = batch['logp']
#         adv = batch['adv']
#         stats = {}
#
#         new_logp, entropy, _ = self.agent.evaluate_actions(obs, act, need_val=False)
#         entropy_scalar = entropy.mean()
#         stats['entropy'] = entropy_scalar.item()
#
#         logratio = new_logp - old_logp
#         ratio = logratio.exp()
#
#         with torch.no_grad():
#             approx_kl =((ratio - 1) - logratio).mean().item()
#             stats['approx_kl'] = approx_kl
#
#         surr1 = ratio * adv
#         surr2 = torch.clamp(ratio, 1.0 - self.cfg.algo.clip_epsilon, 1.0 + self.cfg.algo.clip_epsilon)
#         policy_loss = -torch.min(surr1, surr2).mean()
#
#         return policy_loss, stats
#
#
#     def compute_L_joint(self, obs, v_targ, dist_old):
#         #1. Compute L_aux
#         policy_val = self.agent.get_policy_value(obs).flatten()
#         L_aux = 0.5 * ((policy_val - v_targ)**2).mean()
#
#         dist_new, _  = self.agent.model(obs)
#         kl_divergence = torch.distributions.kl_divergence(dist_old, dist_new).mean()
#
#         L_joint = L_aux + self.cfg.algo.beta_clone * kl_divergence
#
#         return L_joint
#
#
# ######################### VALUE RELATED ##################################################
#
#     def update_value_functions(self):
#         update_stats = defaultdict(list)
#
#         data = self.buffer.get()
#         obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
#             data['obs'],
#             data['act'],
#             data['logp'],
#             data['target_value_r'],
#             data['target_value_c'],
#             data['adv_r'],
#             data['adv_c'],
#         )
#
#         dataloader = DataLoader(
#             dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
#             batch_size=self.cfg.algo.batch_size,
#             shuffle=True,
#         )
#
#         for (obs, act, logp, target_value_r, target_value_c, adv_r, adv_c,) in dataloader:
#             # update reward critic and store loss
#             v_loss_item = self._update_reward_critic(obs, target_value_r)
#             update_stats["v_loss"].append(v_loss_item)
#
#             # optionally update cost critic and store loss
#             if self.cfg.algo.use_cost:
#                 c_loss_item = self._update_cost_critic(obs, target_value_c)
#                 update_stats["c_loss"].append(c_loss_item)
#
#         avg_stats = {k: np.mean(v) for k, v in update_stats.items()}
#         return avg_stats
#
#
#     def compute_L_value_clipped(self, batch):
#         obs = batch['obs']
#         old_val = batch["val"]
#         ret = batch['ret']
#
#         new_val = self.agent.get_value(obs)
#
#         v_loss_unclipped = (new_val - ret) ** 2
#         v_clipped = old_val + torch.clamp(
#             new_val - old_val, -self.cfg.algo.clip_epsilon, self.cfg.algo.clip_epsilon,
#         )
#         v_loss_clipped = (v_clipped - ret) ** 2
#         v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
#
#         return v_loss, {"v_loss": v_loss.item()}
#
#
#     def compute_L_value_unclipped(self, obs, v_targ):
#         val = self.agent.get_value(obs)
#         L_value = 0.5 * ((val -v_targ)**2).mean()
#         return L_value
#
#
# ###### TO MAKE ABC HAPPY:
#
#     def compute_loss(self, batch):
#         pass
