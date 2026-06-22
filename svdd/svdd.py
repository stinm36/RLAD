import json
import os
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class C_AutoEncoder(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, z_dim=4):
        super().__init__()
        self.z_dim = z_dim
        inp = state_dim + action_dim

        # Encoder
        self.fc1 = nn.Linear(inp,        hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=1e-4, affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=1e-4, affine=False)
        self.fc3 = nn.Linear(hidden_dim, z_dim,      bias=False)

        # Decoder
        self.fc4 = nn.Linear(z_dim,      hidden_dim, bias=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim, eps=1e-4, affine=False)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn4 = nn.BatchNorm1d(hidden_dim, eps=1e-4, affine=False)
        self.fc6 = nn.Linear(hidden_dim, inp,        bias=False)

    def encoder(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        return self.fc3(x)

    def decoder(self, x):
        x = F.leaky_relu(self.bn3(self.fc4(x)))
        x = F.leaky_relu(self.bn4(self.fc5(x)))
        return self.fc6(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def _weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


class _SVDDNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, z_dim=4):
        super().__init__()
        inp = state_dim + action_dim
        self.fc1 = nn.Linear(inp,        hidden_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=1e-4, affine=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=1e-4, affine=False)
        self.fc3 = nn.Linear(hidden_dim, z_dim,      bias=False)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = F.leaky_relu(self.bn2(self.fc2(x)))
        return self.fc3(x)


class TrainerDeepSVDD:
    """Train and load a Deep SVDD anomaly detector on (state, action) pairs.

    Weights are saved to:
        {ad_save_path}/svdd/{env}/pretrained_ae.pth   — AE init weights + center
        {ad_save_path}/svdd/{env}/model{tag}.pth      — trained SVDD network
        {ad_save_path}/svdd/{env}/metadata.json        — hyperparameters & timestamp
    """

    def __init__(self, args, state_dim, action_dim, data, device):
        self.args       = args
        self.train_loader = data
        self.device     = device
        self.state_dim  = state_dim
        self.action_dim = action_dim

    # ── Internal path helpers ─────────────────────────────────────────────

    def _save_dir(self):
        d = os.path.join(self.args.ad_save_path, 'svdd', self.args.env)
        os.makedirs(d, exist_ok=True)
        return d

    def _pretrain_path(self):
        return os.path.join(self._save_dir(), 'pretrained_ae.pth')

    def _model_path(self):
        tag = getattr(self.args, 'svdd_tag', '')
        suffix = f'_{tag}' if tag else ''
        return os.path.join(self._save_dir(), f'model{suffix}.pth')

    def _metadata_path(self):
        return os.path.join(self._save_dir(), 'metadata.json')

    def _save_metadata(self, stage):
        meta = {
            'env':        self.args.env,
            'ad_module':  'svdd',
            'stage':      stage,
            'hidden_dim': self.args.hidden_dim,
            'latent_dim': self.args.latent_dim,
            'epochs_ad':  self.args.epochs_ad,
            'lr_ad':      self.args.lr_ad,
            'weight_decay_ae':   self.args.weight_decay_ae,
            'weight_decay_svdd': self.args.weight_decay_svdd,
            'svdd_tag':   getattr(self.args, 'svdd_tag', ''),
            'timestamp':  datetime.datetime.now().isoformat(timespec='seconds'),
        }
        with open(self._metadata_path(), 'w') as f:
            json.dump(meta, f, indent=2)

    # ── AE Pre-training ───────────────────────────────────────────────────

    def _set_center(self, model, eps=0.1):
        model.eval()
        zs = []
        with torch.no_grad():
            for x in self.train_loader:
                z = model.encoder(x.float().to(self.device))
                zs.append(z.detach())
        c = torch.cat(zs).mean(dim=0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] =  eps
        return c

    def pretrain(self):
        """Train the autoencoder, then save its encoder weights as SVDD init."""
        ae = C_AutoEncoder(
            self.state_dim, self.action_dim,
            self.args.hidden_dim, self.args.latent_dim
        ).to(self.device)
        ae.apply(_weights_init_normal)

        optimizer = optim.Adam(
            ae.parameters(),
            lr=self.args.lr_ad,
            weight_decay=self.args.weight_decay_ae,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        ae.train()
        for epoch in range(self.args.epochs_ad):
            total_loss = 0.0
            for x in self.train_loader:
                x = x.float().to(self.device)
                optimizer.zero_grad()
                x_hat = ae(x)
                loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if epoch % 10 == 0:
                print(f'[SVDD pretrain] Epoch {epoch:4d} | '
                      f'Loss: {total_loss / len(self.train_loader):.6f}')

        # Save encoder weights and hypersphere center
        c = self._set_center(ae)
        net = _SVDDNetwork(
            self.state_dim, self.action_dim,
            self.args.hidden_dim, self.args.latent_dim
        ).to(self.device)
        net.load_state_dict(ae.state_dict(), strict=False)
        torch.save(
            {'center': c.cpu().tolist(), 'net_dict': net.state_dict()},
            self._pretrain_path(),
        )
        print(f'[SVDD pretrain] Saved to {self._pretrain_path()}')

    # ── SVDD Training ─────────────────────────────────────────────────────

    def train(self):
        """Train Deep SVDD, starting from pretrained AE encoder weights."""
        net = _SVDDNetwork(
            self.state_dim, self.action_dim,
            self.args.hidden_dim, self.args.latent_dim
        ).to(self.device)

        pretrain_path = self._pretrain_path()
        if os.path.exists(pretrain_path):
            ckpt = torch.load(pretrain_path)
            net.load_state_dict(ckpt['net_dict'])
            c = torch.tensor(ckpt['center'], device=self.device)
        else:
            print('[SVDD train] No pretrained weights found; using random init.')
            net.apply(_weights_init_normal)
            c = torch.randn(self.args.latent_dim, device=self.device)

        optimizer = optim.Adam(
            net.parameters(),
            lr=self.args.lr_ad,
            weight_decay=self.args.weight_decay_svdd,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        n_epochs = getattr(self.args, 'svdd_train_epochs', 0) or self.args.epochs_ad
        net.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for x in self.train_loader:
                x = x.float().to(self.device)
                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if epoch % 10 == 0:
                print(f'[SVDD train] Epoch {epoch:4d} | '
                      f'Loss: {total_loss / len(self.train_loader):.6f}')

        torch.save(net.state_dict(), self._model_path())
        self._save_metadata(stage='train_complete')
        print(f'[SVDD train] Saved to {self._model_path()}')
        self.net = net
        self.c   = c

    # ── Inference ─────────────────────────────────────────────────────────

    def load_SVDD(self):
        """Load trained SVDD network and hypersphere center for inference."""
        ckpt = torch.load(self._pretrain_path())
        c    = torch.tensor(ckpt['center'])

        net = _SVDDNetwork(
            self.state_dim, self.action_dim,
            self.args.hidden_dim, self.args.latent_dim,
        )
        net.load_state_dict(torch.load(self._model_path()))
        return net, c
