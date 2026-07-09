import numpy as np
import torch
import torch.nn as nn
import logging, os, datetime
from torch import optim
from torch.nn import Parameter
from tqdm import trange, tqdm

class Trainer:
    def __init__(self, model, device, dataloader, args):
        self.model = model
        self.device = device
        self.args = args
        self.dataloader = dataloader
        self.criterion = nn.BCEWithLogitsLoss()
        self.lr = args.lr
        self.batch = args.batch_size_ad
        self.b_prime = Parameter(torch.Tensor(self.batch, args.dim).to(self.device))
        torch.nn.init.xavier_normal_(self.b_prime)
        self.optim = optim.Adam(
            list(self.model.parameters()) + [self.b_prime],
            lr = self.lr, betas = (0.5, 0.999)
        )

    def train(self):
        # Log & Save

        log_dir = './DSEBM/saves/' + self.args.env

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        if not os.path.exists(os.path.join(self.args.ad_save_path, self.args.env)):
            os.makedirs(os.path.join(self.args.ad_save_path, self.args.env))

        logger = logging.getLogger(name=self.args.env)
        logger.setLevel(logging.INFO) ## 경고 수준 설정
        formatter = logging.Formatter('|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S'
                                    )
        file_handler = logging.FileHandler(os.path.join(log_dir, 'train.log')) ## 파일 핸들러 생성
        file_handler.setFormatter(formatter) ## 텍스트 포맷 설정
        logger.addHandler(file_handler) ## 핸들러 등록

        logger.propagate = False

        scheduler = optim.lr_scheduler.MultiStepLR(self.optim, 
                    milestones=self.args.lr_milestones, gamma=0.1)

        for epoch in range(self.args.epochs_ad):
            print(f"\nEpoch : {epoch + 1} of {self.args.epochs_ad}")
            losses = 0
            energies = 0
            with trange(len(self.dataloader)) as t:
                for i, data in enumerate(self.dataloader):
                    x = data.to(self.device)
                    noise = self.model.random_noise(data).to(self.device)

                    x_noise = x + noise
                    x.requires_grad_()
                    x_noise.requires_grad_()

                    out = self.model(x)
                    out_noise = self.model(x_noise)

                    energy = self.energy(x, out)
                    energy_noise = self.energy(x_noise, out_noise)

                    dEn_dX = torch.autograd.grad(energy_noise, x_noise, retain_graph=True, create_graph=True)
                    fx_noise = (x_noise - dEn_dX[0])
                    loss = self.loss(x, fx_noise)

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    losses += loss.item()
                    energies += energy.item()
                    losses /= (i+1)
                    energies /= (i+1)
                    t.set_postfix(
                        l='{:05.4f}'.format(loss),
                        e='{:05.4f}'.format(energy),

                    )
                    t.update()
            if epoch % 10 == 0 :
                logger.info(f'Training DSEBM... Epoch : {epoch}, Loss : {loss}, Energy : {energy}')
                logger.info(f'Model checkpoint saved at epoch {epoch}')
                torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'energy': energy.item(),
                'b_prime': self.b_prime.data
                }, os.path.join(self.args.ad_save_path, self.args.env, 'checkpoint.pth'))
                        


    def energy(self, x, y):
        return 0.5 * torch.sum(torch.square(x - self.b_prime)) - torch.sum(y)

    def energy_per_sample(self, x):
        """Per-sample DSEBM energy. Returns shape (N,).

            E(x) = 0.5 * ||x - b'||^2 - sum_j f(x)_j ,   f = self.model

        Unlike `energy` (which sums over the whole batch into a single scalar
        for the autograd-based training loss), this keeps one energy per point
        — required for per-(state,action) scoring: penalty and score plots.

        `self.b_prime` is stored as (batch, dim); DSEBM's b' is a single
        (dim,) data-space bias, so we collapse the batch-tied axis by averaging.
        """
        b_prime = self.b_prime
        if b_prime.dim() == 2:
            b_prime = b_prime.mean(dim=0)               # (dim,)
        y = self.model(x)                               # (N, D)
        recon = 0.5 * torch.square(x - b_prime).sum(dim=-1)   # (N,)
        potential = y.sum(dim=-1)                       # (N,)
        return recon - potential                        # (N,)

    def loss(self, X, fx_noise):
        out = torch.square(X - fx_noise)
        out = torch.sum(out, dim=-1)
        out = torch.mean(out)
        return out
    
    def load_ckpt(self):
        # dsebm = DSEBM(obs_dim, action_dim, hidden_dim = args.hidden_dim).to(device)
        sd = torch.load(os.path.join(self.args.ad_save_path, self.args.env, 'checkpoint.pth'))
        self.model.load_state_dict(sd['model_state_dict'])
        # Restore the learned data-space bias b'; without this, energy scoring
        # uses the random xavier init from __init__ and is wrong.
        if 'b_prime' in sd:
            bp = sd['b_prime'].to(self.device)
            if bp.shape == self.b_prime.shape:
                self.b_prime.data = bp
            else:
                # checkpoint batch size differs from current — resize then copy
                self.b_prime = Parameter(bp)
        self.model.eval()   # BatchNorm/dropout-free here, but keep inference mode explicit
        # return dsebm
