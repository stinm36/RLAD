"""Implicit Q-Learning (Kostrikov et al. 2021)
https://arxiv.org/abs/2110.06169

Adapted for compatibility with this rlkit fork.
Changes vs. the original iql.py:
  - Removed missing imports (add_prefix, LinearTransform, np_to_pytorch_batch, pickle, time)
  - Replaced dist = policy(obs) API with TanhGaussianPolicy.log_prob(obs, actions)
  - Removed self.replay_buffer._size reference
  - Simplified reward/terminal transform to identity by default
"""
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule


class IQLTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            vf,
            quantile=0.7,
            target_qf1=None,
            target_qf2=None,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=3e-4,
            qf_lr=3e-4,
            policy_weight_decay=0,
            q_weight_decay=0,
            optimizer_class=optim.Adam,

            policy_update_period=1,
            q_update_period=1,

            clip_score=100.0,
            soft_target_tau=5e-3,
            target_update_period=1,
            beta=3.0,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.vf = vf

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            weight_decay=policy_weight_decay,
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            weight_decay=q_weight_decay,
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self.q_update_period = q_update_period
        self.policy_update_period = policy_update_period
        self.clip_score = clip_score
        self.beta = beta
        self.quantile = quantile

    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        QF Loss  (Bellman backup via V instead of max_a Q)
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        target_vf_pred = self.vf(next_obs).detach()

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_vf_pred
        q_target = q_target.detach()
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """
        VF Loss  (expectile regression)
        """
        q_pred = torch.min(
            self.target_qf1(obs, actions),
            self.target_qf2(obs, actions),
        ).detach()
        vf_pred = self.vf(obs)
        vf_err = vf_pred - q_pred
        vf_sign = (vf_err > 0).float()
        vf_weight = (1 - vf_sign) * self.quantile + vf_sign * (1 - self.quantile)
        vf_loss = (vf_weight * (vf_err ** 2)).mean()

        """
        Policy Loss  (advantage-weighted regression)
        """
        adv = q_pred - vf_pred.detach()
        exp_adv = torch.exp(adv / self.beta)
        if self.clip_score is not None:
            exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        # log π(a | s) for the dataset actions — uses TanhGaussianPolicy.log_prob
        policy_logpp = self.policy.log_prob(obs, actions)          # (B,)
        weights = exp_adv[:, 0].detach()                           # (B,)
        policy_loss = (-policy_logpp * weights).mean()

        """
        Update networks
        """
        if self._n_train_steps_total % self.q_update_period == 0:
            self.qf1_optimizer.zero_grad()
            qf1_loss.backward()
            self.qf1_optimizer.step()

            self.qf2_optimizer.zero_grad()
            qf2_loss.backward()
            self.qf2_optimizer.step()

            self.vf_optimizer.zero_grad()
            vf_loss.backward()
            self.vf_optimizer.step()

        if self._n_train_steps_total % self.policy_update_period == 0:
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        """
        Logging
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['VF Loss']  = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q1 Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Q2 Predictions', ptu.get_numpy(q2_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions',  ptu.get_numpy(vf_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Q Targets',      ptu.get_numpy(q_target)))
            self.eval_statistics.update(create_stats_ordered_dict('Advantage Weights', ptu.get_numpy(weights)))
            self.eval_statistics.update(create_stats_ordered_dict('rewards',  ptu.get_numpy(rewards)))
            self.eval_statistics.update(create_stats_ordered_dict('terminals', ptu.get_numpy(terminals)))

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.vf,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            vf=self.vf,
        )
