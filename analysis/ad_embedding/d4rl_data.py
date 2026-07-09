"""Lightweight D4RL dataset loader — bypasses gym / mujoco-py / d4rl.

The repo's `svdd.dataset.D4RLDataset` does `gym.make(env) + d4rl.qlearning_dataset()`,
which pulls in the full mujoco-py + d4rl stack. For AD training / score visualisation
we only need the raw (observation, action) arrays, so we read the D4RL HDF5 files
directly with h5py.

Datasets are mirrored on the Hugging Face hub (the original Berkeley `rail` host is
frequently down). File naming there is `{task}_{quality}-v2.hdf5`, e.g.
`hopper_medium_expert-v2.hdf5`.
"""

import os
import urllib.request

import h5py
import numpy as np
import torch

HF_BASE = "https://huggingface.co/datasets/imone/D4RL/resolve/main"

# Default cache dir. Override with $SH_D4RL_DIR. Datasets are ~150MB-2GB each,
# so prefer the big local disk /data2 (root '/' runs full); fall back to the
# repo's data/ dir if /data2 isn't available.
def _default_data_dir():
    env = os.environ.get("SH_D4RL_DIR")
    if env:
        return env
    # /data2 root is root-owned; the per-user subdir /data2/sohyung is writable.
    big = "/data2/sohyung/d4rl"
    big_parent = os.path.dirname(big)
    if os.path.isdir(big_parent) and os.access(big_parent, os.W_OK):
        return big
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        "data", "d4rl")


DATA_DIR = _default_data_dir()


def _hf_filename(env_name):
    """`hopper-medium-expert-v2` -> `hopper_medium_expert-v2.hdf5`.

    D4RL env ids use '-' between task and quality words; the HF mirror uses '_'
    for everything except the trailing version tag (`-v2`).
    """
    assert env_name.endswith(("-v0", "-v1", "-v2")), env_name
    base, ver = env_name.rsplit("-", 1)          # ('hopper-medium-expert', 'v2')
    return f"{base.replace('-', '_')}-{ver}.hdf5"


def download(env_name, data_dir=DATA_DIR):
    """Download the HDF5 for `env_name` if not already cached. Returns local path."""
    os.makedirs(data_dir, exist_ok=True)
    fname = _hf_filename(env_name)
    dst = os.path.join(data_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return dst
    url = f"{HF_BASE}/{fname}"
    tmp = dst + ".part"
    print(f"[d4rl_data] downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    os.rename(tmp, dst)
    print(f"[d4rl_data] saved {dst} ({os.path.getsize(dst) / 1e6:.1f} MB)")
    return dst


def load_state_action(env_name, data_dir=DATA_DIR, as_tensor=True):
    """Return concatenated (state, action) array for `env_name`.

    Shape: (N, obs_dim + action_dim), float32.
    """
    path = download(env_name, data_dir)
    with h5py.File(path, "r") as f:
        obs = np.asarray(f["observations"], dtype=np.float32)
        act = np.asarray(f["actions"], dtype=np.float32)
    sa = np.concatenate([obs, act], axis=1)
    if as_tensor:
        return torch.from_numpy(sa)
    return sa


def dims(env_name, data_dir=DATA_DIR):
    """Return (obs_dim, action_dim) without loading everything into memory."""
    path = download(env_name, data_dir)
    with h5py.File(path, "r") as f:
        return f["observations"].shape[1], f["actions"].shape[1]


def load_transitions(env_name, data_dir=DATA_DIR):
    """Return full offline transitions as a dict of np.float32 arrays.

    Keys match rlkit's GPUReplayBuffer.add_path: observations, actions,
    rewards, next_observations, terminals. The D4RL v2 HDF5s already store
    next_observations, so no shifting is needed. `terminals` uses the true
    done flag only (NOT timeouts) — a timeout is an episode cut, not a real
    terminal, so bootstrapping should continue across it.
    """
    path = download(env_name, data_dir)
    with h5py.File(path, "r") as f:
        d = dict(
            observations=np.asarray(f["observations"], dtype=np.float32),
            actions=np.asarray(f["actions"], dtype=np.float32),
            rewards=np.asarray(f["rewards"], dtype=np.float32).reshape(-1),
            next_observations=np.asarray(f["next_observations"], dtype=np.float32),
            terminals=np.asarray(f["terminals"], dtype=np.float32).reshape(-1),
        )
    return d


if __name__ == "__main__":
    import sys
    env = sys.argv[1] if len(sys.argv) > 1 else "hopper-medium-v2"
    sa = load_state_action(env, as_tensor=False)
    o, a = dims(env)
    print(f"{env}: (s,a) array {sa.shape}  | obs_dim={o} act_dim={a}  dtype={sa.dtype}")
