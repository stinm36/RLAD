import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.gpu_replay_buffer import GPUReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql_ad import CQLADTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import numpy as np
import torch
from torch.utils.data import DataLoader

from svdd.dataset import D4RLDataset
from svdd.svdd import TrainerDeepSVDD
from dagmm.solver import Solver
from fAnogan.model import Generator, Discriminator, Encoder
from fAnogan.training import train_wgangp, train_encoder_izif
from DSEBM.model import DSEBM
from DSEBM.trainer import Trainer as DSEBMTrainer
from mahal.mahal import MahalanobisAD
from gmm.gmm import GPUGMM
from maf.maf import MAFAD

import json
import os
import argparse
import gym
import d4rl
import datetime


def experiment(args, variant):
    if torch.cuda.is_available():
        print("GPU IS AVAILABLE!")
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env

    args.obs_dim    = obs_dim    = expl_env.observation_space.low.size
    args.action_dim = action_dim = eval_env.action_space.low.size

    dataset   = D4RLDataset(args.env)
    dataloader = DataLoader(dataset, batch_size=args.batch_size_ad, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True,
                            persistent_workers=True, prefetch_factor=4)

    # AD module setup (identical to RLAD.py)
    if args.ad_module == 'svdd':
        ad = TrainerDeepSVDD(args, obs_dim, action_dim, dataloader, device)
        if args.ad_train:
            ad.pretrain()
            ad.train()
    elif args.ad_module == 'dagmm':
        ad = Solver(obs_dim + action_dim, dataloader, args, device)
        if args.ad_train:
            ad.train()
    elif args.ad_module == 'dsebm':
        args.lr  = args.lr_ad
        args.dim = obs_dim + action_dim
        ad_model = DSEBM(obs_dim, action_dim, hidden_dim=args.hidden_dim).to(device)
        ad = DSEBMTrainer(ad_model, device, dataloader, args)
        if args.ad_train:
            ad.train()
    elif args.ad_module == 'fanogan':
        generator     = Generator(args.obs_dim, args.action_dim, latent_dim=args.fanogan_latent_dim)
        discriminator = Discriminator(args.obs_dim, args.action_dim, hidden_dim=args.hidden_dim)
        encoder       = Encoder(args.obs_dim, args.action_dim, latent_dim=args.fanogan_latent_dim,
                                hidden_dim=args.hidden_dim)
        if args.ad_train:
            train_wgangp(args, generator, discriminator, dataloader, device, lambda_gp=10)
            train_encoder_izif(args, generator, discriminator, encoder, dataloader, device, kappa=1.0)
        ad_dir = os.path.join(args.ad_save_path, args.ad_module, args.env)
        statedict         = torch.load(os.path.join(ad_dir, 'gan.pth'))
        statedict_encoder = torch.load(os.path.join(ad_dir, 'encoder.pth'))
        generator.load_state_dict(statedict['Generator'])
        discriminator.load_state_dict(statedict['Discriminator'])
        encoder.load_state_dict(statedict_encoder)
        generator.to(device).eval()
        discriminator.to(device).eval()
        encoder.to(device).eval()
        ad = {'Generator': generator, 'Discriminator': discriminator,
              'Encoder': encoder, 'kappa': args.kappa}
    elif args.ad_module == 'mahal':
        ad = MahalanobisAD(device)
        if args.ad_train:
            ad.fit(dataloader)
        else:
            ad.load(os.path.join(args.ad_save_path, args.ad_module, args.env, 'mahal.pth'))
    elif args.ad_module == 'gmm':
        ad = GPUGMM(n_components=args.gmm_k, device=device, n_iter=args.gmm_n_iter)
        if args.ad_train:
            ad.fit(dataloader)
        else:
            ad.load(os.path.join(args.ad_save_path, args.ad_module, args.env, 'gmm.pth'))
    elif args.ad_module == 'maf':
        ad = MAFAD(device, hidden_dim=args.hidden_dim, n_layers=args.maf_n_layers)
        if args.ad_train:
            ad.fit(dataloader, lr=args.lr_ad, epochs=args.epochs_ad)
            ad.save(os.path.join(args.ad_save_path, args.ad_module, args.env, 'maf.pth'))
        else:
            ad.load(os.path.join(args.ad_save_path, args.ad_module, args.env, 'maf.pth'))
    elif args.ad_module == 'random':
        ad = None
    else:
        raise Exception("Wrong Module Name")

    M = variant['layer_size']
    qf1       = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M])
    qf2       = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M])
    target_qf1 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M])
    target_qf2 = FlattenMlp(input_size=obs_dim + action_dim, output_size=1, hidden_sizes=[M, M])
    policy    = TanhGaussianPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=[M, M])

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(eval_env, eval_policy)
    expl_path_collector = MdpPathCollector(expl_env, policy)
    replay_buffer = GPUReplayBuffer(variant['replay_buffer_size'], expl_env)

    trainer = CQLADTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        ad=ad,
        args=args,
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
    parser = argparse.ArgumentParser(description='CQL + AD')

    parser.add_argument("--env",            type=str,      default='hopper-medium-v2')
    parser.add_argument("--gpu",            type=str,      default='0')
    parser.add_argument('--seed',           type=int,      default=int(np.random.randint(0, 100000)))
    parser.add_argument('--all_saves',      type=str,      default='runs',
                        help='Root directory for all experiment logs')
    parser.add_argument('--trial_name',     type=str,      default='',
                        help='Human-readable run label (used in log dir name)')
    parser.add_argument('--nepochs',        type=int,      default=3000)
    parser.add_argument('--qf_lr',          type=float,    default=3e-4)
    parser.add_argument('--policy_lr',      type=float,    default=3e-4)
    parser.add_argument('--batch_size',     type=int,      default=512)

    # AD shared
    parser.add_argument('--ad_module',      type=str,      default='svdd')
    parser.add_argument('--ad_save_path',   type=str,      default='./weights_ad')
    parser.add_argument('--ad_train',       type=str2bool, default=True)
    parser.add_argument('--weight_function',type=str,      default='hill',
                        help='hill/tanh_penalty (recommended for Mode 3) | identity | sigmoid')
    parser.add_argument('--penalty_coef',   type=float,    default=0.01)
    parser.add_argument('--epochs_ad',      type=int,      default=200)
    parser.add_argument('--lr_ad',          type=float,    default=1e-4)
    parser.add_argument('--lr_milestones',  type=list,     default=[50, 150])
    parser.add_argument('--batch_size_ad',  type=int,      default=256)
    parser.add_argument('--hidden_dim',     type=int,      default=256)
    parser.add_argument('--latent_dim',     type=int,      default=128)
    parser.add_argument('--weight_decay_svdd', type=float, default=0.5e-6)
    parser.add_argument('--weight_decay_ae',   type=float, default=0.5e-3)
    parser.add_argument('--patience',       type=int,      default=50)

    # DAGMM
    parser.add_argument('--num_epochs',         type=int,   default=200)
    parser.add_argument('--gmm_k',              type=int,   default=2)
    parser.add_argument('--lambda_energy',      type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag',    type=float, default=0.005)
    parser.add_argument('--dagmm_latent_dim',   type=int,   default=1)
    parser.add_argument('--pretrained_model',   default=None)
    parser.add_argument('--mode',               type=str,   default='train')
    parser.add_argument('--use_tensorboard',    type=str2bool, default=False)
    parser.add_argument('--log_step',           type=int,   default=500)
    parser.add_argument('--sample_step',        type=int,   default=500)
    parser.add_argument('--model_save_step',    type=int,   default=500)

    # fAnoGAN
    parser.add_argument('--fanogan_latent_dim', type=int,   default=100)
    parser.add_argument('--fanogan_n_critic',   type=int,   default=5)
    parser.add_argument('--kappa',              type=float, default=1.0)

    # GMM / MAF
    parser.add_argument('--gmm_n_iter',     type=int,   default=100)
    parser.add_argument('--maf_n_layers',   type=int,   default=5)

    # CQL-specific
    parser.add_argument('--min_q_version',      type=int,      default=3)
    parser.add_argument('--min_q_weight',       type=float,    default=5.0)
    parser.add_argument('--temp',               type=float,    default=1.0)
    parser.add_argument('--with_lagrange',      type=str2bool, default=False)
    parser.add_argument('--lagrange_thresh',    type=float,    default=10.0)
    parser.add_argument('--num_random',         type=int,      default=10)
    parser.add_argument('--max_q_backup',       type=str2bool, default=False)
    parser.add_argument('--deterministic_backup', type=str2bool, default=True)

    args = parser.parse_args()

    variant = dict(
        algorithm='RLOCC-CQL',
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
            penalty_coef=args.penalty_coef,
        ),
    )

    ts       = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'{args.trial_name}_{ts}' if args.trial_name else ts
    log_dir  = os.path.join(args.all_saves, args.env, args.ad_module, run_name)

    setup_logger(run_name, variant=variant, base_log_dir=args.all_saves,
                 name=run_name, log_dir=log_dir)

    os.makedirs(log_dir, exist_ok=True)
    config = vars(args).copy()
    config['timestamp'] = ts
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, default=str)
    ptu.set_gpu_mode(True)
    experiment(args, variant)
