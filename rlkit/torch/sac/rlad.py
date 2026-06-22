from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

# AMP helpers — accessed via module attribute to stay compatible with
# PyTorch 1.13.x where the type stubs may be absent in some IDEs.
GradScaler = torch.cuda.amp.GradScaler
autocast   = torch.cuda.amp.autocast

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


def setup_ad(ad, ad_module):
    """Prepare AD model for inference. Returns (ad_model, center_c)."""
    c = None
    if ad_module == 'svdd':
        ad, c = ad.load_SVDD()
        ad.cuda().eval()
        c = c.cuda()
    elif ad_module == 'dagmm':
        ad.dagmm.cuda().eval()
    elif ad_module == 'dsebm':
        ad.load_ckpt()
    # fanogan, mahal, gmm, maf, random: already ready after experiment()
    return ad, c


def make_weight_function(name):
    """Return a callable that maps raw AD score → penalty magnitude.

    ── Penalty mode (Mode 3): high score → HIGH penalty ────────────────────
        hill         : w = score/(score+T)   range [0, 1)  bounded ✓
        tanh_penalty : w = tanh(score/T)     range [0, 1)  bounded ✓
        identity     : w = score             range [0, ∞)  unbounded

    For 'hill'/'tanh_penalty': penalty_coef * w ∈ (0, penalty_coef) always.
    T=5.0 is the temperature; tune to the typical in-distribution AD score.
    For 'identity', softplus is applied automatically in train_from_torch.
    """
    if name == 'hill':
        T = 5.0
        return lambda w: w / (w + T)
    elif name == 'tanh_penalty':
        T = 5.0
        return lambda w: torch.tanh(w / T)
    elif name == 'identity':
        return lambda w: w
    else:
        raise ValueError(
            f"Unknown weight_function: '{name}'. "
            f"Choose from: hill (recommended), tanh_penalty, identity."
        )


def calc_anomaly_score(ad, state_action, c=None, ad_type=None):
    if ad_type == 'svdd':
        return torch.sqrt(torch.sum(
            (ad(state_action) - c) ** 2,
            dim=tuple(range(1, state_action.dim()))
        ))
    elif ad_type == 'dagmm':
        state_action = ad.to_var(state_action)
        _, _, z, _ = ad.dagmm(state_action)
        weight, _ = ad.dagmm.compute_energy(z, size_average=False)
        return weight
    elif ad_type == 'dsebm':
        out = ad.model(state_action)
        return ad.energy(state_action, out)
    elif ad_type == 'fanogan':
        criterion = nn.MSELoss()
        g = ad['Generator']
        d = ad['Discriminator']
        e = ad['Encoder']
        real_z = e(state_action)
        fake_data = g(real_z)
        real_features = d.forward_features(state_action)
        fake_features = d.forward_features(fake_data)
        return criterion(fake_data, state_action) + ad['kappa'] * criterion(fake_features, real_features)
    elif ad_type in ('mahal', 'gmm', 'maf'):
        return ad.score(state_action)
    elif ad_type == 'random':
        return torch.rand(state_action.shape[0], device=state_action.device)
    else:
        raise ValueError(f"Unknown ad_type: '{ad_type}'")


class SACTrainer(TorchTrainer):
    """SAC + Mode-3 AD penalty on Q-targets.

    Q-target = r + γ·min(Q1',Q2') - α·log π - penalty_coef · g(score(s', π(s')))

    where g = weight_function maps the raw AD anomaly score to a bounded penalty.
    OOD-leading (s', a') pairs receive a larger penalty, driving structural pessimism
    that propagates backward through the Bellman backup chain.

    Bound: ||Q^π - Q_pen^π||_∞ ≤ penalty_coef / (1 - γ)

    Performance notes
    -----------------
    • When paired with GPUReplayBuffer, the batch dict contains GPU tensors and
      TorchTrainer.train() skips the numpy→GPU conversion entirely.
    • QF1 and QF2 share a single backward pass (no retain_graph overhead).
    • Mixed precision (AMP) is enabled when use_amp=True (default).
    """

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

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,

            weight_function='hill',
            weight_actor=False,
            penalty_coef=0.01,
            use_amp=True,
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
        self.weight_actor = weight_actor
        self.discrete = False  # required by BatchRLAlgorithm

        # Mixed precision
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler  = GradScaler(enabled=self.use_amp)

        # Entropy tuning
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            self.target_entropy = (target_entropy if target_entropy
                                   else -np.prod(self.env.action_space.shape).item())
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(self.policy.parameters(), lr=policy_lr)
        self.qf1_optimizer    = optimizer_class(self.qf1.parameters(),    lr=qf_lr)
        self.qf2_optimizer    = optimizer_class(self.qf2.parameters(),    lr=qf_lr)

        self.discount     = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

        self._current_epoch       = 0
        self._num_q_update_steps  = 0
        self._num_policy_update_steps = 0

        # AD setup
        self.c = None
        if args.ad_module == 'svdd':
            self.ad, self.c = ad.load_SVDD()
            self.ad.cuda().eval()
            self.c = self.c.cuda()
        elif args.ad_module == 'dagmm':
            self.ad = ad
            self.ad.dagmm.cuda().eval()
        elif args.ad_module == 'dsebm':
            self.ad = ad
            self.ad.load_ckpt()
        elif args.ad_module in ('fanogan', 'mahal', 'gmm', 'maf'):
            self.ad = ad
        elif args.ad_module == 'random':
            self.ad = None
        else:
            raise ValueError(f"Unknown ad_module: '{args.ad_module}'")

        self.weight_function = make_weight_function(weight_function)
        print(f'[SACTrainer] AD module: {args.ad_module} | '
              f'weight_function: {weight_function} | '
              f'penalty_coef: {penalty_coef} | '
              f'AMP: {self.use_amp}')

    def eval_q_custom(self, custom_policy, data_batch, q_function=None):
        """Called by BatchRLAlgorithm for Q-value evaluation logging."""
        if q_function is None:
            q_function = self.qf1
        obs = data_batch['observations']
        new_obs_actions, *_ = self.policy(obs, reparameterize=True, return_log_prob=True)
        return float(q_function(obs, new_obs_actions).mean().detach().cpu().numpy())

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards   = batch['rewards']
        terminals = batch['terminals']
        obs       = batch['observations']
        actions   = batch['actions']
        next_obs  = batch['next_observations']

        # ── Policy and Alpha Loss ─────────────────────────────────────────
        # autocast: forward passes run in FP16 on tensor-core GPUs;
        # log/exp/sqrt auto-promoted to FP32 by PyTorch AMP.
        with autocast(enabled=self.use_amp):
            new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
                obs, reparameterize=True, return_log_prob=True,
            )

            if self.use_automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()          # log_alpha stays FP32; no scaler needed
                self.alpha_optimizer.step()
                alpha = self.log_alpha.exp()
            else:
                alpha_loss = 0
                alpha = 1

            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions),
            )

            if self.weight_actor:
                with torch.no_grad():
                    sa_curr = torch.cat([obs, new_obs_actions], dim=-1)
                    # AD score: keep FP32 for numerical stability
                    with autocast(enabled=False):
                        score_curr = calc_anomaly_score(
                            self.ad, sa_curr.float(), c=self.c, ad_type=self.args.ad_module)
                    actor_weight = torch.exp(-score_curr).unsqueeze(-1)  # (B, 1)
                policy_loss = ((alpha * log_pi - q_new_actions) * actor_weight).mean()
            else:
                actor_weight = None
                policy_loss = (alpha * log_pi - q_new_actions).mean()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        self.scaler.scale(policy_loss).backward()
        self.scaler.step(self.policy_optimizer)

        # ── Q-function Loss (Mode 3 — Penalised-reward MDP) ──────────────
        #
        #   q_target = r + γ·min(Q1',Q2') - α·log π(a'|s')
        #              - penalty_coef · g(score(s', a'))
        #
        #   Bound: ||Q^π - Q_pen^π||_∞ ≤ penalty_coef/(1-γ)
        #   QF1 and QF2 share a single backward (no retain_graph overhead).
        # ─────────────────────────────────────────────────────────────────
        with autocast(enabled=self.use_amp):
            q1_pred = self.qf1(obs, actions)
            q2_pred = self.qf2(obs, actions)

            new_next_actions, _, _, new_log_pi, *_ = self.policy(
                next_obs, reparameterize=True, return_log_prob=True,
            )
            target_q_values = torch.min(
                self.target_qf1(next_obs, new_next_actions),
                self.target_qf2(next_obs, new_next_actions),
            ) - alpha * new_log_pi

        with torch.no_grad():
            sa_next = torch.cat([next_obs, new_next_actions.detach()], dim=-1)
            # AD score in FP32 for numerical stability
            with autocast(enabled=False):
                score = calc_anomaly_score(
                    self.ad, sa_next.float(), c=self.c, ad_type=self.args.ad_module)
            weight = self.weight_function(score)

        if self.args.weight_function == 'identity':
            penalty = self.penalty_coef * torch.nn.functional.softplus(weight)
        else:
            penalty = self.penalty_coef * weight   # bounded to (0, penalty_coef)

        q_target = (self.reward_scale * rewards
                    + (1. - terminals) * self.discount * target_q_values
                    - penalty.unsqueeze(-1)).detach()

        with autocast(enabled=self.use_amp):
            qf1_loss = (q1_pred - q_target).pow(2).mean()
            qf2_loss = (q2_pred - q_target).pow(2).mean()
            # Sum: single backward traverse, no retain_graph overhead.
            # Gradients are independent (qf1 ∩ qf2 params = ∅).
            qf_loss = qf1_loss + qf2_loss

        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        self.scaler.scale(qf_loss).backward()
        self.scaler.step(self.qf1_optimizer)
        self.scaler.step(self.qf2_optimizer)

        # GradScaler update — called once per train step
        self.scaler.update()

        # ── Soft Target Updates ───────────────────────────────────────────
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        # ── Logging (one batch per epoch) ─────────────────────────────────
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            self.eval_statistics['QF1 Loss']       = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss']       = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss']    = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics['AD Weight Mean'] = weight.mean().item()
            self.eval_statistics['AD Penalty Mean'] = penalty.mean().item()
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions', ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions', ptu.get_numpy(q2_pred)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets', ptu.get_numpy(q_target)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis', ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu', ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std', ptu.get_numpy(policy_log_std)))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha']      = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
            if self.weight_actor and actor_weight is not None:
                self.eval_statistics['Actor Weight Mean'] = actor_weight.mean().item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [self.policy, self.qf1, self.qf2, self.target_qf1, self.target_qf2]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
