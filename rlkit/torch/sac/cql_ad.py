"""CQL + Anomaly Detection penalty on Q-targets.

Integrates AD the same way as rlad.py (theoretically mode 3):
    q_target = r + γ Q(s', π(s')) − penalty_coef · softplus(score(s', π(s')))
then adds the standard CQL conservatism term on top.
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


class CQLADTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            ad,
            args,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class=optim.Adam,

            soft_target_tau=5e-3,
            target_update_period=1,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=5.0,
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,

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
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.args = args
        self.penalty_coef = penalty_coef

        # AD setup
        self.ad, self.c = setup_ad(ad, args.ad_module)
        self.weight_function = make_weight_function(args.weight_function)

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = (target_entropy if target_entropy
                                   else -np.prod(self.env.action_space.shape).item())
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class([self.log_alpha_prime], lr=qf_lr)

        self.qf_criterion = nn.MSELoss()
        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer   = optimizer_class(self.qf1.parameters(),   lr=qf_lr)
        self.qf2_optimizer   = optimizer_class(self.qf2.parameters(),   lr=qf_lr)

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start
        self._current_epoch = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0

        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight
        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=self.temp, threshold=20)
        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random
        self.discrete = False

    def _get_tensor_values(self, obs, actions, network=None):
        num_repeat = int(actions.shape[0] / obs.shape[0])
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        preds = network(obs_temp, actions)
        return preds.view(obs.shape[0], num_repeat, 1)

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(obs_temp, reparameterize=True, return_log_prob=True)
        return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards   = batch['rewards']
        terminals = batch['terminals']
        obs       = batch['observations']
        actions   = batch['actions']
        next_obs  = batch['next_observations']

        """Policy and Alpha Loss"""
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(self.qf1(obs, new_obs_actions), self.qf2(obs, new_obs_actions))
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        if self._current_epoch < self.policy_eval_start:
            policy_log_prob = self.policy.log_prob(obs, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

        """QF Loss"""
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(next_obs, reparameterize=True, return_log_prob=True)

        if not self.max_q_backup:
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            )
            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        else:
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            tqf1 = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
            tqf2 = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(tqf1, tqf2)

        # AD penalty on Bellman target
        with torch.no_grad():
            sa = torch.cat([next_obs, new_next_actions], dim=-1)
            weight = calc_anomaly_score(self.ad, sa, c=self.c, ad_type=self.args.ad_module)
            weight = self.weight_function(weight).unsqueeze(-1)  # (B, 1)

        if self.args.weight_function == 'identity':
            ad_penalty = self.penalty_coef * torch.nn.functional.softplus(weight)
        else:
            ad_penalty = self.penalty_coef * weight  # already bounded
        q_target = (self.reward_scale * rewards
                    + (1. - terminals) * self.discount * target_q_values
                    - ad_penalty)
        q_target = q_target.detach()

        qf1_loss = self.qf_criterion(q1_pred, q_target)
        qf2_loss = self.qf_criterion(q2_pred, q_target)

        """CQL conservatism term"""
        random_actions_tensor = torch.FloatTensor(
            q2_pred.shape[0] * self.num_random, actions.shape[-1]
        ).uniform_(-1, 1).to(obs.device)

        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs, num_actions=self.num_random, network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs, num_actions=self.num_random, network=self.policy)

        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor.detach(), network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor.detach(), network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor.detach(), network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor.detach(), network=self.qf2)

        if self.min_q_version == 3:
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat([q1_rand - random_density,
                                 q1_next_actions - new_log_pis.detach(),
                                 q1_curr_actions - curr_log_pis.detach()], 1)
            cat_q2 = torch.cat([q2_rand - random_density,
                                 q2_next_actions - new_log_pis.detach(),
                                 q2_curr_actions - curr_log_pis.detach()], 1)
        else:
            cat_q1 = torch.cat([q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1)
            cat_q2 = torch.cat([q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1)

        min_qf1_loss = (torch.logsumexp(cat_q1 / self.temp, dim=1).mean() * self.min_q_weight * self.temp
                        - q1_pred.mean() * self.min_q_weight)
        min_qf2_loss = (torch.logsumexp(cat_q2 / self.temp, dim=1).mean() * self.min_q_weight * self.temp
                        - q2_pred.mean() * self.min_q_weight)

        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1e6)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)
            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        """Update networks"""
        # Policy first: policy_loss uses q_new_actions through qf1/qf2.
        # Q params must not be modified before policy backward.
        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        self.policy_optimizer.step()

        # Single backward: qf1 and qf2 have disjoint parameters → no retain_graph needed.
        # When with_lagrange=True, alpha_prime_loss.backward(retain_graph=True) above
        # keeps the shared graph alive; (qf1_loss + qf2_loss).backward() consumes it.
        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        (qf1_loss + qf2_loss).backward()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        """Logging"""
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF1 Loss']     = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss']     = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
            self.eval_statistics['min QF2 Loss'] = np.mean(ptu.get_numpy(min_qf2_loss))
            self.eval_statistics['Policy Loss']  = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics['AD Weight Mean'] = weight.mean().item()
            self.eval_statistics.update(create_stats_ordered_dict('Q1 Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Q2 Predictions', ptu.get_numpy(q2_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Q Targets',      ptu.get_numpy(q_target)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis',        ptu.get_numpy(log_pi)))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]

    def get_snapshot(self):
        return dict(policy=self.policy, qf1=self.qf1, qf2=self.qf2,
                    target_qf1=self.target_qf1, target_qf2=self.target_qf2)
