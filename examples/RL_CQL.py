import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import numpy as np
import torch
import os
import argparse
import gym
import d4rl
import datetime


def experiment(args, variant):
    if torch.cuda.is_available():
        print("GPU IS AVAILABLE!")

    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    obs_dim    = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M])
    qf2 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M])
    target_qf1 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M])
    target_qf2 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M])
    policy = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=[M, M])

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    expl_path_collector = MdpPathCollector(expl_env, policy)
    replay_buffer = EnvReplayBuffer(variant['replay_buffer_size'], expl_env)

    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs'],
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs'],
    )
    algorithm.to(ptu.device)
    algorithm.train()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CQL')
    parser.add_argument("--env",           type=str,   default='hopper-medium-v2')
    parser.add_argument("--gpu",           type=str,   default='0')
    parser.add_argument('--seed',          type=int,   default=int(np.random.randint(0, 100000)))
    parser.add_argument('--all_saves',     type=str,   default='saves_cql')
    parser.add_argument('--trial_name',    type=str,   default='')
    parser.add_argument('--log_dir',       type=str,   default='./default/')
    parser.add_argument('--nepochs',       type=int,   default=3000)
    parser.add_argument('--qf_lr',         type=float, default=3e-4)
    parser.add_argument('--policy_lr',     type=float, default=3e-4)
    parser.add_argument('--batch_size',    type=int,   default=512)
    # CQL-specific
    parser.add_argument('--min_q_version', type=int,   default=3)
    parser.add_argument('--min_q_weight',  type=float, default=5.0)
    parser.add_argument('--temp',          type=float, default=1.0)
    parser.add_argument('--with_lagrange', type=str2bool, default=False)
    parser.add_argument('--lagrange_thresh', type=float, default=10.0)
    parser.add_argument('--num_random',    type=int,   default=10)
    parser.add_argument('--max_q_backup',  type=str2bool, default=False)
    parser.add_argument('--deterministic_backup', type=str2bool, default=True)

    args = parser.parse_args()

    variant = dict(
        algorithm='CQL',
        env_name=args.env,
        layer_size=256,
        replay_buffer_size=int(1e6),
        algorithm_kwargs=dict(
            num_epochs=args.nepochs,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=args.batch_size,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=1,
            min_q_version=args.min_q_version,
            min_q_weight=args.min_q_weight,
            temp=args.temp,
            with_lagrange=args.with_lagrange,
            lagrange_thresh=args.lagrange_thresh,
            num_random=args.num_random,
            max_q_backup=args.max_q_backup,
            deterministic_backup=args.deterministic_backup,
        ),
    )

    file_name = args.trial_name
    setup_logger(
        file_name, variant=variant, base_log_dir=args.all_saves, name=file_name,
        log_dir=os.path.join(
            args.all_saves, args.log_dir, file_name, args.env,
            datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        ),
    )
    ptu.set_gpu_mode(True)
    experiment(args, variant)
