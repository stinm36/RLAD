import os
import torch
from tqdm import tqdm


class MahalanobisAD:
    """
    Anomaly detection via Mahalanobis distance.

    Fits a single multivariate Gaussian to the training data and scores
    samples by their squared Mahalanobis distance from the mean.
    Entirely GPU-native via PyTorch — no training loop, just a one-pass
    statistics computation over the dataset.
    """

    def __init__(self, device):
        self.device = device
        self.mean = None     # (D,)
        self.cov_inv = None  # (D, D)

    def fit(self, dataloader, reg=1e-6):
        print("Fitting Mahalanobis AD (computing mean and covariance)...")
        chunks = [x.float() for x in tqdm(dataloader, desc="Loading data")]
        data = torch.cat(chunks, dim=0).to(self.device)  # (N, D)
        N, D = data.shape

        self.mean = data.mean(0)                                      # (D,)
        centered = data - self.mean                                   # (N, D)
        cov = centered.T @ centered / N                               # (D, D)
        reg_cov = cov + reg * torch.eye(D, device=self.device)
        self.cov_inv = torch.linalg.inv(reg_cov)                     # (D, D)
        print(f"Mahalanobis AD fitted on ({N}, {D}) dataset.")

    def score(self, x):
        """Squared Mahalanobis distance. Input: (B, D), Output: (B,)"""
        diff = x - self.mean                                          # (B, D)
        return (diff @ self.cov_inv * diff).sum(-1)                  # (B,)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'mean': self.mean.cpu(), 'cov_inv': self.cov_inv.cpu()}, path)
        print(f"Saved Mahalanobis parameters → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.mean = ckpt['mean'].to(self.device)
        self.cov_inv = ckpt['cov_inv'].to(self.device)
        print(f"Loaded Mahalanobis parameters ← {path}")
