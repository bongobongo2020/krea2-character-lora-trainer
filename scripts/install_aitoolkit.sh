#!/bin/bash
# Clone + install AI Toolkit (ostris) and its Python venv using uv.
# Invoked by the "Install AI Toolkit" button in the web UI (or run directly).
# This is the Turbo-native engine: it trains on krea/Krea-2-Turbo with the
# de-distill assistant LoRA. Idempotent: safe to re-run.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AITK="$ROOT/ai-toolkit"

echo "==> Target: $AITK"

# 1. Clone (or update) the repo.
if [ ! -d "$AITK/.git" ]; then
  echo "==> Cloning AI Toolkit (ostris)…"
  git clone https://github.com/ostris/ai-toolkit.git "$AITK"
  git -C "$AITK" submodule update --init --recursive || true
else
  echo "==> Already cloned, pulling latest…"
  git -C "$AITK" pull --ff-only || true
  git -C "$AITK" submodule update --init --recursive || true
fi

cd "$AITK"

# 2. Create the venv if needed (prefer uv; fall back to python venv).
USE_UV=0
if command -v uv >/dev/null 2>&1; then USE_UV=1; fi

if [ ! -d "venv" ]; then
  echo "==> Creating venv (Python 3.12)…"
  if [ "$USE_UV" = "1" ]; then
    uv venv --python 3.12 venv
  else
    (python3.12 -m venv venv) 2>/dev/null || python3 -m venv venv
  fi
fi

pip_install() {
  if [ "$USE_UV" = "1" ]; then
    uv pip install --python "$AITK/venv/bin/python" "$@"
  else
    "$AITK/venv/bin/pip" install "$@"
  fi
}

# 3. PyTorch stack (CUDA 12.4). torchaudio is required by AI Toolkit's config
#    modules — installing the matching pinned trio avoids the "No module named
#    'torchaudio'" failure. Re-running is a no-op once satisfied.
echo "==> Ensuring PyTorch + torchvision + torchaudio (CUDA 12.4)…"
pip_install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 4. AI Toolkit requirements.
echo "==> Installing AI Toolkit requirements…"
pip_install -r requirements.txt

echo "==> Done. AI Toolkit is ready (Krea 2 Turbo + de-distill adapter download on first run)."
