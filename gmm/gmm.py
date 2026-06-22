import os
import numpy as np
import torch
from tqdm import trange, tqdm


class GPUGMM:
    """
    Gaussian Mixture Model (GMM) fitted via EM entirely on GPU.
    Uses negative log-likelihood as anomaly score.

    All covariance matrices are full (not diagonal) and regularised
    with a small identity term for numerical stability.
    The E-step is chunked along N to keep peak GPU memory bounded.
    """

    def __init__(self, n_components=2, device='cpu', n_iter=100, tol=1e-4, reg_covar=1e-6):
        self.K = n_components
        self.device = device
        self.n_iter = n_iter
        self.tol = tol
        self.reg_covar = reg_covar

        self.pi = None       # (K,)      mixing weights
        self.mu = None       # (K, D)    component means
        self.cov = None      # (K, D, D) component covariances
        self.cov_inv = None  # (K, D, D) cached inverses  — set after fit/load
        self.log_det = None  # (K,)      cached log-dets  — set after fit/load

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, dataloader, chunk_size=50000):
        """
        Run EM on the full dataset.
        chunk_size: number of samples per E-step chunk (controls GPU memory).
        """
        print(f"Fitting GMM (K={self.K}) via EM on GPU...")
        chunks = [x.float() for x in tqdm(dataloader, desc="Loading data")]
        data = torch.cat(chunks, dim=0).to(self.device)  # (N, D)
        N, D = data.shape
        print(f"Dataset: ({N}, {D})")

        # Initialise: K random data points as means, identity covariances
        idx = torch.randperm(N, device=self.device)[:self.K]
        self.mu = data[idx].clone()                                                   # (K, D)
        self.pi = torch.full((self.K,), 1.0 / self.K, device=self.device)            # (K,)
        self.cov = torch.eye(D, device=self.device).unsqueeze(0).repeat(self.K, 1, 1)  # (K, D, D)

        prev_ll = float('-inf')
        for it in trange(self.n_iter, desc="EM"):
            # E-step (chunked)
            log_resp = self._e_step(data, chunk_size, D)   # (N, K)

            # Log-likelihood for convergence check
            ll = torch.logsumexp(log_resp, dim=1).mean().item()

            # M-step
            resp = torch.softmax(log_resp, dim=1)          # (N, K) normalised
            N_k = resp.sum(0).clamp(min=1e-8)              # (K,)
            self.pi = N_k / N
            self.mu = resp.T @ data / N_k.unsqueeze(1)    # (K, D)
            for k in range(self.K):
                diff_k = data - self.mu[k]                 # (N, D)
                w = resp[:, k]                             # (N,)
                self.cov[k] = (w.unsqueeze(1) * diff_k).T @ diff_k / N_k[k]

            if abs(ll - prev_ll) < self.tol:
                print(f"\nConverged at iteration {it + 1}  (LL={ll:.4f})")
                break
            prev_ll = ll

        self._update_cache()
        print(f"GMM fitting complete. Final log-likelihood: {prev_ll:.4f}")

    def _e_step(self, data, chunk_size, D):
        """Chunked E-step to control peak GPU memory."""
        reg_inv = self._reg_cov_inv()   # (K, D, D) — computed once per iteration
        ld = self._log_det()            # (K,)
        log_resp_list = []
        for start in range(0, len(data), chunk_size):
            chunk = data[start:start + chunk_size]               # (C, D)
            diff = chunk.unsqueeze(1) - self.mu.unsqueeze(0)    # (C, K, D)
            tmp = torch.einsum('ckd,kde->cke', diff, reg_inv)   # (C, K, D)
            mahal = (tmp * diff).sum(-1)                         # (C, K)
            log_resp_list.append(
                torch.log(self.pi + 1e-10).unsqueeze(0)
                - 0.5 * (D * np.log(2 * np.pi) + ld.unsqueeze(0) + mahal)
            )
        return torch.cat(log_resp_list, dim=0)                   # (N, K)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reg_cov(self):
        D = self.mu.shape[1]
        return self.cov + self.reg_covar * torch.eye(D, device=self.device)

    def _reg_cov_inv(self):
        return torch.linalg.inv(self._reg_cov())

    def _log_det(self):
        return torch.logdet(self._reg_cov())

    def _update_cache(self):
        self.cov_inv = torch.linalg.inv(self._reg_cov())   # (K, D, D)
        self.log_det = torch.logdet(self._reg_cov())        # (K,)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, x):
        """Negative log-likelihood anomaly score. Input: (B, D), Output: (B,)"""
        D = self.mu.shape[1]
        diff = x.unsqueeze(1) - self.mu.unsqueeze(0)                    # (B, K, D)
        tmp = torch.einsum('bkd,kde->bke', diff, self.cov_inv)          # (B, K, D)
        mahal = (tmp * diff).sum(-1)                                     # (B, K)
        log_probs = (
            torch.log(self.pi + 1e-10).unsqueeze(0)
            - 0.5 * (D * np.log(2 * np.pi) + self.log_det.unsqueeze(0) + mahal)
        )
        return -torch.logsumexp(log_probs, dim=1)                        # (B,)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'pi': self.pi.cpu(), 'mu': self.mu.cpu(), 'cov': self.cov.cpu()}, path)
        print(f"Saved GMM parameters → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.pi = ckpt['pi'].to(self.device)
        self.mu = ckpt['mu'].to(self.device)
        self.cov = ckpt['cov'].to(self.device)
        self._update_cache()
        print(f"Loaded GMM parameters ← {path}")
