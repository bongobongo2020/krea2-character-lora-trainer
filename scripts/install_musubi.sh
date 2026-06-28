#!/bin/bash
# Clone + install Musubi-Tuner and the Python 3.11 training virtualenv.
# Invoked by the "Install musubi-tuner" button in the web UI (or run directly).
# Idempotent: safe to re-run (e.g. after a failed dependency install).
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MUSUBI="$ROOT/musubi-tuner"

echo "==> Target: $MUSUBI"

# 1. Clone (or update) the repo.
if [ ! -d "$MUSUBI/.git" ]; then
  echo "==> Cloning Musubi-Tuner (kohya-ss)…"
  git clone https://github.com/kohya-ss/musubi-tuner.git "$MUSUBI"
else
  echo "==> Already cloned, pulling latest…"
  git -C "$MUSUBI" pull --ff-only || true
fi

cd "$MUSUBI"

# 2. Create the venv if needed (but always (re)install deps below).
USE_UV=0
if command -v uv >/dev/null 2>&1; then USE_UV=1; fi

if [ ! -d ".venv" ]; then
  echo "==> Creating training virtualenv (Python 3.11)…"
  if [ "$USE_UV" = "1" ]; then
    uv venv --python 3.11 .venv
  else
    (python3.11 -m venv .venv) 2>/dev/null || python3 -m venv .venv
  fi
fi
source .venv/bin/activate

pip_install() {
  if [ "$USE_UV" = "1" ]; then uv pip install "$@"; else pip install "$@"; fi
}

# 3. PyTorch (CUDA 12.4). Re-running is a no-op once satisfied.
echo "==> Ensuring PyTorch (CUDA 12.4)…"
pip_install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 4. Musubi-Tuner itself + its dependencies. The repo is a pyproject package
#    (there is no requirements.txt), so install it editable.
echo "==> Installing Musubi-Tuner package + dependencies…"
pip_install -e .
pip_install accelerate

echo "==> Done. Musubi-Tuner is ready."
