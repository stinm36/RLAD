import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange, tqdm


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """Linear layer with a fixed binary mask applied to the weight matrix."""
    def __init__(self, in_features, out_features, mask):
        super().__init__(in_features, out_features)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (Germain et al. 2015).

    Given input x ∈ R^D, outputs (mu, log_alpha) ∈ R^D each, such that
    output dimension d only depends on x_1, ..., x_{d-1}.
    This gives a valid autoregressive factorisation of the density.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        D, H = input_dim, hidden_dim

        # Integer orderings ------------------------------------------------
        m_in = torch.arange(1, D + 1)                   # input:  [1, 2, ..., D]
        m_h  = torch.arange(H) % max(1, D - 1) + 1     # hidden: values in [1, D-1]
        m_out = torch.arange(1, D + 1)                  # output: [1, 2, ..., D]

        # Input → hidden: hidden k sees input j if m_h[k] >= m_in[j]
        mask1 = (m_h.unsqueeze(1) >= m_in.unsqueeze(0)).float()    # (H, D)
        # Hidden → output (×2 for mu and log_alpha): output d sees hidden k if m_out[d] > m_h[k]
        mask2 = (m_out.unsqueeze(1) > m_h.unsqueeze(0)).float()    # (D, H)

        self.net = nn.Sequential(
            MaskedLinear(D, H, mask1),
            nn.Tanh(),
            MaskedLinear(H, 2 * D, mask2.repeat(2, 1)),
        )

    def forward(self, x):
        out = self.net(x)
        mu, log_alpha = out.chunk(2, dim=-1)   # each (B, D)
        return mu, log_alpha


class MAFLayer(nn.Module):
    """
    Single MAF coupling layer (Papamakarios et al. 2017).

    Forward  (x → z, parallel over d):  z_d = (x_d − μ_d(x_{<d})) · exp(−α_d(x_{<d}))
    Log Jacobian:                        Σ_d −α_d(x_{<d})
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.made = MADE(input_dim, hidden_dim)

    def forward(self, x):
        mu, log_alpha = self.made(x)
        z = (x - mu) * torch.exp(-log_alpha)
        log_det = -log_alpha.sum(-1)           # (B,)
        return z, log_det


class MAF(nn.Module):
    """
    Composition of MAFLayer blocks with alternating input permutations
    (reverse/identity) to ensure all dimensions interact across layers.
    """
    def __init__(self, input_dim, hidden_dim=256, n_layers=5):
        super().__init__()
        self.D = input_dim
        self.layers = nn.ModuleList(
            [MAFLayer(input_dim, hidden_dim) for _ in range(n_layers)]
        )
        # Register alternating permutations as buffers
        for i in range(n_layers - 1):
            perm = (torch.arange(input_dim - 1, -1, -1) if i % 2 == 0
                    else torch.arange(input_dim))
            self.register_buffer(f'perm_{i}', perm)
        self._n_perms = n_layers - 1

    def log_prob(self, x):
        """Compute log p(x) under the MAF. Fully parallel over dimensions."""
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        for i, layer in enumerate(self.layers):
            x, log_det = layer(x)
            log_det_total = log_det_total + log_det
            if i < self._n_perms:
                x = x[:, getattr(self, f'perm_{i}')]
        # Base distribution: N(0, I)
        log_pz = -0.5 * (x.pow(2) + np.log(2 * np.pi)).sum(-1)
        return log_pz + log_det_total


# ---------------------------------------------------------------------------
# Anomaly-detection wrapper
# ---------------------------------------------------------------------------

class MAFAD:
    """
    Anomaly detection via Masked Autoregressive Flow.

    Trains a MAF on the offline dataset by maximising log-likelihood.
    Anomaly score = negative log-likelihood: −log p(s, a).
    Input data is z-score normalised before being fed to the flow.
    """

    def __init__(self, device, hidden_dim=256, n_layers=5):
        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.model = None
        self._mean = None   # (D,) normalisation stats
        self._std  = None   # (D,)

    def fit(self, dataloader, lr=1e-3, epochs=100):
        print(f"Fitting MAF (layers={self.n_layers}, hidden={self.hidden_dim})...")
        chunks = [x.float() for x in tqdm(dataloader, desc="Loading data")]
        all_data = torch.cat(chunks, dim=0)   # (N, D) on CPU
        N, D = all_data.shape
        print(f"Dataset: ({N}, {D})")

        # Compute normalisation stats on CPU, store on device
        self._mean = all_data.mean(0).to(self.device)
        self._std  = all_data.std(0).clamp(min=1e-8).to(self.device)

        # Normalise on CPU, then build a loader that moves to GPU per batch
        all_data_norm = (all_data - self._mean.cpu()) / self._std.cpu()
        dataset = torch.utils.data.TensorDataset(all_data_norm)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=512, shuffle=True, pin_memory=True
        )

        self.model = MAF(D, self.hidden_dim, self.n_layers).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for epoch in trange(epochs, desc="MAF training"):
            total_loss = 0.0
            for (x,) in loader:
                x = x.to(self.device)
                loss = -self.model.log_prob(x).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}  NLL: {total_loss/len(loader):.4f}")

        self.model.eval()
        print("MAF fitting complete.")

    def score(self, x):
        """Negative log-likelihood anomaly score. Input: (B, D), Output: (B,)"""
        x_norm = (x - self._mean) / self._std
        return -self.model.log_prob(x_norm)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model':  self.model.state_dict(),
            'mean':   self._mean.cpu(),
            'std':    self._std.cpu(),
            'config': {
                'input_dim':  self.model.D,
                'hidden_dim': self.hidden_dim,
                'n_layers':   self.n_layers,
            },
        }, path)
        print(f"Saved MAF → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        cfg  = ckpt['config']
        self.model      = MAF(cfg['input_dim'], cfg['hidden_dim'], cfg['n_layers']).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self._mean      = ckpt['mean'].to(self.device)
        self._std       = ckpt['std'].to(self.device)
        self.hidden_dim = cfg['hidden_dim']
        self.n_layers   = cfg['n_layers']
        print(f"Loaded MAF ← {path}")
