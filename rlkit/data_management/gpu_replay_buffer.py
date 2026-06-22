"""GPU-resident replay buffer.

All transition tensors live in CUDA memory permanently.
Sampling uses torch.randint on-device — zero CPU↔GPU roundtrips during training.

Memory footprint (float32):
  1M transitions, obs_dim=11, action_dim=3 → ~108 MB
  1M transitions, obs_dim=376 (humanoid), action_dim=17 → ~1.6 GB

Drop-in replacement for EnvReplayBuffer / SimpleReplayBuffer.
Returns GPU tensor dicts from random_batch(), so TorchTrainer skips
the usual np_to_pytorch_batch() conversion.
"""
import numpy as np
import torch

from rlkit.data_management.replay_buffer import ReplayBuffer


class GPUReplayBuffer(ReplayBuffer):

    def __init__(self, max_replay_buffer_size, env, device=None):
        from gym.spaces import Discrete
        from rlkit.envs.env_utils import get_dim

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        obs_dim    = get_dim(env.observation_space)
        action_dim = get_dim(env.action_space)
        self._obs_dim    = obs_dim
        self._action_dim = action_dim
        self.max_size    = max_replay_buffer_size

        # Allocate all buffers on GPU upfront
        self._obs       = torch.zeros(max_replay_buffer_size, obs_dim,    device=device)
        self._next_obs  = torch.zeros(max_replay_buffer_size, obs_dim,    device=device)
        self._actions   = torch.zeros(max_replay_buffer_size, action_dim, device=device)
        self._rewards   = torch.zeros(max_replay_buffer_size, 1,          device=device)
        self._terminals = torch.zeros(max_replay_buffer_size, 1,          device=device)

        self._top  = 0
        self._size = 0

        # Handle discrete action spaces (one-hot)
        from gym.spaces import Discrete as GymDiscrete
        self._discrete_actions = isinstance(env.action_space, GymDiscrete)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _t(self, x):
        """numpy/scalar → GPU float32 tensor, zero-copy when possible."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32, non_blocking=True)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    # ── ReplayBuffer interface ────────────────────────────────────────────

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        if self._discrete_actions:
            one_hot = np.zeros(self._action_dim, dtype=np.float32)
            one_hot[action] = 1.0
            action = one_hot

        i = self._top
        self._obs[i]       = self._t(observation)
        self._actions[i]   = self._t(action)
        self._rewards[i]   = float(reward)
        self._next_obs[i]  = self._t(next_observation)
        self._terminals[i] = float(terminal)

        self._top  = (self._top + 1) % self.max_size
        self._size = min(self._size + 1, self.max_size)

    def add_path(self, path):
        """Vectorised bulk insertion — avoids per-step Python overhead."""
        obs       = np.asarray(path['observations'],      dtype=np.float32)
        actions   = np.asarray(path['actions'],           dtype=np.float32)
        rewards   = np.asarray(path['rewards'],           dtype=np.float32).reshape(-1, 1)
        next_obs  = np.asarray(path['next_observations'], dtype=np.float32)
        terminals = np.asarray(path['terminals'],         dtype=np.float32).reshape(-1, 1)
        n = len(obs)

        # Compute insertion indices (wraps around the ring buffer)
        idxs = torch.arange(self._top, self._top + n, device=self.device) % self.max_size

        # Single async H→D transfer per field
        self._obs[idxs]       = torch.from_numpy(obs).to(self.device, non_blocking=True)
        self._actions[idxs]   = torch.from_numpy(actions).to(self.device, non_blocking=True)
        self._rewards[idxs]   = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        self._next_obs[idxs]  = torch.from_numpy(next_obs).to(self.device, non_blocking=True)
        self._terminals[idxs] = torch.from_numpy(terminals).to(self.device, non_blocking=True)

        self._top  = (self._top + n) % self.max_size
        self._size = min(self._size + n, self.max_size)
        self.terminate_episode()

    def terminate_episode(self):
        pass

    def random_batch(self, batch_size):
        """Sample a batch on-device — returns GPU tensor dict (no CPU roundtrip)."""
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return {
            'observations':      self._obs[idx],
            'actions':           self._actions[idx],
            'rewards':           self._rewards[idx],
            'terminals':         self._terminals[idx],
            'next_observations': self._next_obs[idx],
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return {'size': self._size}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        pass
