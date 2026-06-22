import torch
import numpy as np
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

import os
import argparse
import gym
import d4rl
import datetime


def experiment(args):
    env = gym.make(args.env)

    if torch.cuda.is_available():
        print("GPU IS AVAILABLE!")
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    dataset_svdd = D4RLDataset(args.env)
    dataloader = DataLoader(dataset_svdd, batch_size=args.batch_size, shuffle=True, num_workers = 4, pin_memory = True)


    if args.ad_module == 'svdd':
        ad = TrainerDeepSVDD(args, obs_dim, action_dim, dataloader, device)
        if args.ad_train:
            ad.pretrain()
            ad.train()
    elif args.ad_module == 'dagmm':
        ad = Solver(obs_dim+action_dim, dataloader, args, device)
        if args.ad_train:
            ad.train()
    elif args.ad_module == 'dsebm':
        args.lr = args.lr_ad
        args.dim = obs_dim + action_dim
        ad_model = DSEBM(obs_dim, action_dim, hidden_dim = args.hidden_dim).to(device)
        ad = DSEBMTrainer(ad_model, device, dataloader, args)
        if args.ad_train:
            ad.train()
    elif args.ad_module == 'fanogan':
        generator = Generator(args.obs_dim, args.action_dim, latent_dim=args.fanogan_latent_dim)
        discriminator = Discriminator(args.obs_dim, args.action_dim, hidden_dim=args.hidden_dim)
        encoder = Encoder(args.obs_dim, args.action_dim, latent_dim=args.fanogan_latent_dim, hidden_dim=args.hidden_dim)
        if args.ad_train:
            train_wgangp(args, generator, discriminator,
                 dataloader, device, lambda_gp=10)
    
            train_encoder_izif(args, generator, discriminator, encoder,
                            dataloader, device, kappa=1.0)
        statedict = torch.load(os.path.join(args.ad_save_path, args.env, 'gan.pth'))
        statedict_encoder = torch.load(os.path.join(args.ad_save_path, args.env, 'encoder.pth'))
        generator.load_state_dict(statedict['Generator'])
        discriminator.load_state_dict(statedict['Discriminator'])
        encoder.load_state_dict(statedict_encoder)
        
        generator.to(device).eval()
        discriminator.to(device).eval()
        encoder.to(device).eval()

        ad = {'Generator' : generator, 'Discriminator' : discriminator, 'Encoder' : encoder, 'kappa' : args.kappa}
    elif args.ad_module == 'mahal':
        ad = MahalanobisAD(device)
        if args.ad_train:
            ad.fit(dataloader)
            ad.save(os.path.join(args.ad_save_path, args.ad_module, args.env, 'mahal.pth'))
    elif args.ad_module == 'gmm':
        ad = GPUGMM(n_components=args.gmm_k, device=device, n_iter=args.gmm_n_iter)
        if args.ad_train:
            ad.fit(dataloader)
            ad.save(os.path.join(args.ad_save_path, args.ad_module, args.env, 'gmm.pth'))
    elif args.ad_module == 'maf':
        ad = MAFAD(device, hidden_dim=args.hidden_dim, n_layers=args.maf_n_layers)
        if args.ad_train:
            ad.fit(dataloader, lr=args.lr_ad, epochs=args.epochs_ad)
            ad.save(os.path.join(args.ad_save_path, args.ad_module, args.env, 'maf.pth'))
    else:
        raise Exception("Wrong Module Name")


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
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(description='AD Pretraining')

    # From BEAR
    parser.add_argument("--env", type=str, default='hopper-medium-v2')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument('--seed', default= int(np.random.randint(0, 100000)), type=int)
    parser.add_argument('--all_saves', default="saves_no_policy_weight_with_penalty", type=str)
    parser.add_argument('--trial_name', default="", type=str)
    parser.add_argument('--log_dir', default='./default/', type=str, 
                        help="Location for logging")
    parser.add_argument('--epochs_ad', default=200, type=int)
    parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--lr_ad', default=1e-3, type=float)
    parser.add_argument('--lr_milestones', default=[50, 150], type=list)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--ad_train', default=True, type=str2bool)
    parser.add_argument('--latent_dim', default=128, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--normal_class', default=1, type=int)
    parser.add_argument('--weight_decay_svdd', default=0.5e-6, type=float)
    parser.add_argument('--weight_decay_ae', default=0.5e-3, type=float)
    parser.add_argument('--batch_size_ad', default=512, type=int)
    parser.add_argument('--ad_module', default="svdd", type=str)
    parser.add_argument('--ad_save_path', default="./weights_ad", type=str)

    # DAGMM hyperparameters
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--gmm_k', type=int, default=2)
    parser.add_argument('--gmm_n_iter', type=int, default=100)
    parser.add_argument('--maf_n_layers', type=int, default=5)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.005)
    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--dagmm_latent_dim', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--sample_step', type=int, default=500)
    parser.add_argument('--model_save_step', type=int, default=500)
    
    # AnoGAN hyperparameters
    parser.add_argument('--anogan_latent_dim', default=100, type=int)

    # f-AnoGAN hyperparameters
    parser.add_argument('--fanogan_latent_dim', default=100, type=int)
    parser.add_argument('--fanogan_n_critic', default=5, type=int)
    parser.add_argument('--kappa', default=1.0, type=float)

    args = parser.parse_args()

    os.makedirs(os.path.join(args.ad_save_path, args.ad_module, args.env), exist_ok=True)
    
    experiment(args)
