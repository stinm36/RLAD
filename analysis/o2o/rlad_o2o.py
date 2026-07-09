"""Offline-to-online (O2O) harness — offline pretrain → online SAC fine-tune.

Three init arms share an IDENTICAL online SAC phase; they differ only in how the
policy/Q are initialised (warm start), isolating init quality (hypothesis H-A):

  ad_rlad : offline pretrain with SAC + AD penalty, then penalty OFF online
  cql     : offline pretrain with CQL, then plain SAC online
  scratch : no offline phase — SAC from random init online

Offline data is read from the D4RL HDF5 via h5py (no d4rl package). The online
env is the matching gym MuJoCo task ({task}-v3), whose obs/act dims equal the
offline data. Run on a GPU node via sbatch (see run_o2o.sbatch).

    python analysis/o2o/rlad_o2o.py --env hopper-medium-v2 --init ad_rlad \
        --offline_epochs 3 --online_epochs 3 --smoke
"""
import argparse, os, sys
from types import SimpleNamespace

import numpy as np
import torch
import gym
import gtimer as gt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.gpu_replay_buffer import GPUReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.torch.sac.rlad import SACTrainer
from rlkit.torch.sac.cql import CQLTrainer

from analysis.ad_embedding import d4rl_data
from svdd.svdd import TrainerDeepSVDD

TASK_ENV = {"hopper": "Hopper-v3", "walker2d": "Walker2d-v3", "halfcheetah": "HalfCheetah-v3"}


def online_env_id(d4rl_env):
    return TASK_ENV[d4rl_env.split("-")[0]]


def make_networks(obs_dim, act_dim, M=256, device="cpu"):
    qf1 = FlattenMlp(input_size=obs_dim + act_dim, output_size=1, hidden_sizes=[M, M])
    qf2 = FlattenMlp(input_size=obs_dim + act_dim, output_size=1, hidden_sizes=[M, M])
    tqf1 = FlattenMlp(input_size=obs_dim + act_dim, output_size=1, hidden_sizes=[M, M])
    tqf2 = FlattenMlp(input_size=obs_dim + act_dim, output_size=1, hidden_sizes=[M, M])
    policy = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=act_dim, hidden_sizes=[M, M])
    nets = [qf1, qf2, tqf1, tqf2, policy]
    for n in nets:
        n.to(device)
    return qf1, qf2, tqf1, tqf2, policy


def ad_args(env, penalty_coef):
    return SimpleNamespace(
        ad_module="svdd", ad_save_path="weights_ad", env=env,
        hidden_dim=256, latent_dim=128, svdd_tag="", svdd_train_epochs=0,
        weight_function="hill", weight_actor=False, penalty_coef=penalty_coef,
        epochs_ad=200, lr_ad=1e-4, lr_milestones=[50, 150],
        weight_decay_ae=0.5e-3, weight_decay_svdd=0.5e-6,
    )


def build_algo(trainer, expl_env, eval_env, expl_pc, eval_pc, buf, *,
               num_epochs, batch_rl, cfg):
    return TorchBatchRLAlgorithm(
        trainer=trainer, exploration_env=expl_env, evaluation_env=eval_env,
        exploration_data_collector=expl_pc, evaluation_data_collector=eval_pc,
        replay_buffer=buf,
        batch_size=cfg.batch_size, max_path_length=cfg.max_path_length,
        num_epochs=num_epochs,
        num_eval_steps_per_epoch=cfg.num_eval_steps_per_epoch,
        num_expl_steps_per_train_loop=(0 if batch_rl else cfg.num_expl_steps_per_train_loop),
        num_trains_per_train_loop=cfg.num_trains_per_train_loop,
        min_num_steps_before_training=(0 if batch_rl else cfg.min_expl_before),
        batch_rl=batch_rl,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="hopper-medium-v2")
    ap.add_argument("--init", choices=["ad_rlad", "cql", "scratch"], default="ad_rlad")
    ap.add_argument("--offline_epochs", type=int, default=250)
    ap.add_argument("--online_epochs", type=int, default=250)
    ap.add_argument("--penalty_coef", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs_o2o")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes for a quick end-to-end check")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ptu.set_gpu_mode(torch.cuda.is_available())
    device = ptu.device
    print(f"[o2o] env={a.env} init={a.init} device={device} "
          f"offline_ep={a.offline_epochs} online_ep={a.online_epochs} smoke={a.smoke}")

    cfg = SimpleNamespace(
        batch_size=256, max_path_length=1000,
        num_eval_steps_per_epoch=1000,
        num_expl_steps_per_train_loop=1000,
        num_trains_per_train_loop=(200 if a.smoke else 1000),
        min_expl_before=1000,
    )

    env_id = online_env_id(a.env)
    expl_env = gym.make(env_id); eval_env = gym.make(env_id)
    obs_dim = expl_env.observation_space.low.size
    act_dim = expl_env.action_space.low.size
    print(f"[o2o] online env {env_id}  obs={obs_dim} act={act_dim}")

    qf1, qf2, tqf1, tqf2, policy = make_networks(obs_dim, act_dim, device=device)
    eval_pc = MdpPathCollector(eval_env, MakeDeterministic(policy))
    expl_pc = MdpPathCollector(expl_env, policy)

    buf = GPUReplayBuffer(int(2e6), expl_env)
    if a.init != "scratch":
        d = d4rl_data.load_transitions(a.env)
        buf.add_path(d)
        print(f"[o2o] offline buffer loaded: {buf._size} transitions")

    # ── build offline trainer for the chosen arm ──
    if a.init == "ad_rlad":
        args = ad_args(a.env, a.penalty_coef)
        ad = TrainerDeepSVDD(args, obs_dim, act_dim, data=None, device=device)
        trainer = SACTrainer(
            env=eval_env, policy=policy, qf1=qf1, qf2=qf2,
            target_qf1=tqf1, target_qf2=tqf2, ad=ad, args=args,
            weight_function="hill", weight_actor=False, penalty_coef=a.penalty_coef,
            policy_lr=3e-4, qf_lr=3e-4, discount=0.99, soft_target_tau=5e-3,
        )
    elif a.init == "cql":
        trainer = CQLTrainer(
            env=eval_env, policy=policy, qf1=qf1, qf2=qf2,
            target_qf1=tqf1, target_qf2=tqf2,
            policy_lr=3e-4, qf_lr=3e-4, discount=0.99, soft_target_tau=5e-3,
        )
    else:  # scratch — plain SAC (no AD, penalty 0), no offline phase
        args = ad_args(a.env, 0.0); args.ad_module = "random"
        trainer = SACTrainer(
            env=eval_env, policy=policy, qf1=qf1, qf2=qf2,
            target_qf1=tqf1, target_qf2=tqf2, ad=None, args=args,
            weight_function="hill", weight_actor=False, penalty_coef=0.0,
            policy_lr=3e-4, qf_lr=3e-4, discount=0.99, soft_target_tau=5e-3,
        )

    exp_name = f"o2o_{a.env}_{a.init}_s{a.seed}"
    setup_logger(exp_name, variant=vars(a), base_log_dir=a.out, name=exp_name,
                 log_dir=os.path.join(a.out, exp_name))

    # ── Phase 1: offline pretrain (skip for scratch) ──
    if a.init != "scratch":
        print("[o2o] === Phase 1: offline pretrain ===")
        gt.reset()   # gtimer global state is per-process; reset before each train loop
        algo1 = build_algo(trainer, expl_env, eval_env, expl_pc, eval_pc, buf,
                           num_epochs=a.offline_epochs, batch_rl=True, cfg=cfg)
        algo1.to(ptu.device); algo1.train()

    # ── switch penalty OFF for the shared online phase (H-A) ──
    if a.init == "ad_rlad":
        trainer.penalty_coef = 0.0
        print("[o2o] penalty OFF for online phase")

    # ── Phase 2: online SAC fine-tune (identical across arms) ──
    print("[o2o] === Phase 2: online SAC fine-tune ===")
    gt.reset()   # clear Phase-1 stamp registry so the online loop's stamps don't collide
    algo2 = build_algo(trainer, expl_env, eval_env, expl_pc, eval_pc, buf,
                       num_epochs=a.online_epochs, batch_rl=False, cfg=cfg)
    algo2.to(ptu.device); algo2.train()
    print("[o2o] DONE")


if __name__ == "__main__":
    main()
