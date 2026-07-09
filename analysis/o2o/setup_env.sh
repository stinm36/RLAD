#!/usr/bin/env bash
# One-shot environment setup for the RLAD offline-to-online (O2O) experiments.
# Reproduces the recipe validated on greenbeard; works on any Ampere+ GPU (A100
# included). Idempotent-ish: safe to re-run.
#
#   bash analysis/o2o/setup_env.sh [ENV_PREFIX] [TORCH_INDEX]
#     ENV_PREFIX   conda env location   (default: ./envs/sh_rlad_o2o)
#     TORCH_INDEX  pytorch wheel index  (default: cu121 — good for A100;
#                                        greenbeard/Blackwell used cu130)
set -euo pipefail

ENV_PREFIX="${1:-./envs/sh_rlad_o2o}"
TORCH_INDEX="${2:-https://download.pytorch.org/whl/cu121}"
MUJOCO_DIR="$HOME/.mujoco/mujoco210"

echo "==> conda env at $ENV_PREFIX (python 3.10 + GL libs for mujoco-py)"
conda create -p "$ENV_PREFIX" -c conda-forge -y \
  python=3.10 glew glfw mesalib patchelf

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

echo "==> mujoco210 binaries"
if [ ! -d "$MUJOCO_DIR" ]; then
  mkdir -p "$HOME/.mujoco"
  curl -sL https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    -o /tmp/mujoco210.tar.gz
  tar -xzf /tmp/mujoco210.tar.gz -C "$HOME/.mujoco/"
fi

echo "==> python deps (order matters: build deps → torch → gym/mujoco)"
pip install "cython<3" "numpy==1.23.5"
pip install torch --index-url "$TORCH_INDEX"
pip install "gym==0.23.1" "mujoco-py>=2.1,<2.2" \
  tensorboard gtimer matplotlib python-dateutil h5py scipy scikit-learn "protobuf<5"

echo "==> persist mujoco env vars in activate.d"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/mujoco_env.sh" <<'EOF'
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:$CONDA_PREFIX/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}
EOF

echo "==> trigger mujoco-py compile + sanity check"
conda deactivate; conda activate "$ENV_PREFIX"
python - <<'PY'
import mujoco_py, gym, torch
e = gym.make("Hopper-v3"); e.reset(); e.step(e.action_space.sample())
print("mujoco_py", mujoco_py.__version__, "| Hopper-v3 obs",
      e.observation_space.shape[0], "act", e.action_space.shape[0])
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY

echo "==> DONE. Activate with: conda activate $ENV_PREFIX"
echo "    (GPU check must run under an allocated GPU, e.g. via srun/sbatch)"
