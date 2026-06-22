"""BEAR + Anomaly Detection penalty on Q-targets.

Standard BEAR Q-target (soft clipped double Q over 10 sampled actions):
    target_Q = 0.75·min(Q1,Q2) + 0.25·max(Q1,Q2)

AD penalty is applied before adding to the Bellman backup:
    target_Q -= penalty_coef · softplus(score(s', π(s')))

where π(s') is one sample from the current actor (not the VAE).
The MMD support constraint on the actor is unchanged.
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.torch.sac.rlad import calc_anomaly_score, setup_ad, make_weight_function


class BEARADTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            vae,
            ad,
            args,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class=optim.Adam,

            soft_target_tau=5e-3,
            target_update_period=1,

            # BEAR
            mode='auto',
            kernel_choice='laplacian',
            policy_update_style=0,
            mmd_sigma=10.0,
            target_mmd_thresh=0.05,
            num_samples_mmd_match=4,

            # AD
            penalty_coef=0.01,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.vae = vae
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.args = args
        self.penalty_coef = penalty_coef

        # AD setup
        self.ad, self.c = setup_ad(ad, args.ad_module)
        self.weight_function = make_weight_function(args.weight_function)

        self.qf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer    = optimizer_class(self.qf1.parameters(),    lr=qf_lr)
        self.qf2_optimizer    = optimizer_class(self.qf2.parameters(),    lr=qf_lr)
        self.vae_optimizer    = optimizer_class(self.vae.parameters(),    lr=3e-4)

        self.mode = mode
        if self.mode == 'auto':
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=1e-3)

        self.mmd_sigma = mmd_sigma
        self.kernel_choice = kernel_choice
        self.num_samples_mmd_match = num_samples_mmd_match
        self.policy_update_style = policy_update_style
        self.target_mmd_thresh = target_mmd_thresh

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._current_epoch = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0

    # ------------------------------------------------------------------
    # MMD kernels
    # ------------------------------------------------------------------

    def mmd_loss_laplacian(self, samples1, samples2, sigma=0.2):
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)
        kxx = torch.mean((-(diff_x_x.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))
        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        kxy = torch.mean((-(diff_x_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))
        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)
        kyy = torch.mean((-(diff_y_y.abs()).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))
        return (kxx + kyy - 2.0 * kxy + 1e-6).sqrt()

    def mmd_loss_gaussian(self, samples1, samples2, sigma=0.2):
        diff_x_x = samples1.unsqueeze(2) - samples1.unsqueeze(1)
        kxx = torch.mean((-(diff_x_x.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))
        diff_x_y = samples1.unsqueeze(2) - samples2.unsqueeze(1)
        kxy = torch.mean((-(diff_x_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))
        diff_y_y = samples2.unsqueeze(2) - samples2.unsqueeze(1)
        kyy = torch.mean((-(diff_y_y.pow(2)).sum(-1) / (2.0 * sigma)).exp(), dim=(1, 2))
        return (kxx + kyy - 2.0 * kxy + 1e-6).sqrt()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards   = batch['rewards']
        terminals = batch['terminals']
        obs       = batch['observations']
        actions   = batch['actions']
        next_obs  = batch['next_observations']

        """VAE behavioral cloning"""
        recon, mean, std = self.vae(obs, actions)
        recon_loss = self.qf_criterion(recon, actions)
        kl_loss    = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss   = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        """Critic Training"""
        with torch.no_grad():
            state_rep  = next_obs.unsqueeze(1).repeat(1, 10, 1).view(next_obs.shape[0] * 10, next_obs.shape[1])
            action_rep = self.policy(state_rep)[0]

            target_qf1 = self.target_qf1(state_rep, action_rep)
            target_qf2 = self.target_qf2(state_rep, action_rep)
            target_Q   = (0.75 * torch.min(target_qf1, target_qf2)
                          + 0.25 * torch.max(target_qf1, target_qf2))
            target_Q   = target_Q.view(next_obs.shape[0], -1).max(1)[0].view(-1, 1)

            # AD penalty: score on (next_obs, one policy sample)
            new_next_a = self.policy(next_obs)[0]           # (B, action_dim)
            sa = torch.cat([next_obs, new_next_a], dim=-1)
            weight = calc_anomaly_score(self.ad, sa, c=self.c, ad_type=self.args.ad_module)
            weight = self.weight_function(weight).unsqueeze(-1)   # (B, 1)
            # identity: softplus compresses heavy tails of unbounded scores.
            # Bounded functions (hill, tanh_penalty): use weight directly —
            # softplus on (0,1) input collapses the dynamic range.
            if self.args.weight_function == 'identity':
                ad_penalty = self.penalty_coef * torch.nn.functional.softplus(weight)
            else:
                ad_penalty = self.penalty_coef * weight

            target_Q = (self.reward_scale * rewards
                        + (1.0 - terminals) * self.discount * target_Q
                        - ad_penalty)

        qf1_pred = self.qf1(obs, actions)
        qf2_pred = self.qf2(obs, actions)
        qf1_loss = (qf1_pred - target_Q.detach()).pow(2).mean()
        qf2_loss = (qf2_pred - target_Q.detach()).pow(2).mean()

        # Single backward: qf1 and qf2 have disjoint parameters → no retain_graph needed
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        """Actor Training (MMD support constraint)"""
        sampled_actions, raw_sampled_actions = self.vae.decode_multiple(
            obs, num_decode=self.num_samples_mmd_match)
        actor_samples, _, _, _, _, _, _, raw_actor_actions = self.policy(
            obs.unsqueeze(1).repeat(1, self.num_samples_mmd_match, 1)
                .view(-1, obs.shape[1]), return_log_prob=True)
        actor_samples    = actor_samples.view(obs.shape[0], self.num_samples_mmd_match, actions.shape[1])
        raw_actor_actions = raw_actor_actions.view(obs.shape[0], self.num_samples_mmd_match, actions.shape[1])

        if self.kernel_choice == 'laplacian':
            mmd_loss = self.mmd_loss_laplacian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)
        else:
            mmd_loss = self.mmd_loss_gaussian(raw_sampled_actions, raw_actor_actions, sigma=self.mmd_sigma)

        q_val1 = self.qf1(obs, actor_samples[:, 0, :])
        q_val2 = self.qf2(obs, actor_samples[:, 0, :])
        q_val  = torch.min(q_val1, q_val2)[:, 0]

        if self._n_train_steps_total >= 40000:
            if self.mode == 'auto':
                policy_loss = (-q_val + self.log_alpha.exp() * (mmd_loss - self.target_mmd_thresh)).mean()
            else:
                policy_loss = (-q_val + 100 * mmd_loss).mean()
        else:
            if self.mode == 'auto':
                policy_loss = (self.log_alpha.exp() * (mmd_loss - self.target_mmd_thresh)).mean()
            else:
                policy_loss = (100 * mmd_loss).mean()

        if self.mode == 'auto':
            self.alpha_optimizer.zero_grad()
            (-policy_loss).backward(retain_graph=True)
            self.alpha_optimizer.step()
            self.log_alpha.data.clamp_(min=-5.0, max=10.0)

        self.policy_optimizer.zero_grad()
        if self.mode == 'auto':
            policy_loss.backward()
        self.policy_optimizer.step()

        """Soft Updates"""
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        """Logging"""
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF1 Loss']   = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss']   = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics['AD Weight Mean'] = weight.mean().item()
            self.eval_statistics.update(create_stats_ordered_dict('Q1 Predictions', ptu.get_numpy(qf1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Q2 Predictions', ptu.get_numpy(qf2_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Q Targets',      ptu.get_numpy(target_Q)))
            self.eval_statistics.update(create_stats_ordered_dict('MMD Loss',       ptu.get_numpy(mmd_loss)))
            if self.mode == 'auto':
                self.eval_statistics['Alpha'] = self.log_alpha.exp().item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2, self.vae]

    def get_snapshot(self):
        return dict(policy=self.policy, qf1=self.qf1, qf2=self.qf2,
                    target_qf1=self.target_qf1, target_qf2=self.target_qf2, vae=self.vae)
