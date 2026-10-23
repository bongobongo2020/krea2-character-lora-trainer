#!/bin/bash
# Clone KohakuMossTrainer and install its Python venv (kohakumoss[cuda] + a
# matching PyTorch build). Invoked by the "Install KohakuMossTrainer" button in
# the web UI (or run directly). Idempotent: safe to re-run.
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOSS="$ROOT/KohakuMossTrainer"

echo "==> Target: $MOSS"

# 1. Clone (or update) the repo. The venv's installed `kohakumoss` package (not
#    this checkout's scripts/) is what actually runs training — the clone just
#    gives us a stable place for the venv to live.
if [ ! -d "$MOSS/.git" ]; then
  echo "==> Cloning KohakuMossTrainer…"
  git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer.git "$MOSS"
else
  echo "==> Already cloned, pulling latest…"
  git -C "$MOSS" pull --ff-only || true
fi

cd "$MOSS"

# 2. Create the venv if needed (prefer uv; fall back to python venv).
USE_UV=0
if command -v uv >/dev/null 2>&1; then USE_UV=1; fi

if [ ! -d ".venv" ]; then
  echo "==> Creating venv (Python 3.12)…"
  if [ "$USE_UV" = "1" ]; then
    uv venv --python 3.12 .venv
  else
    (python3.12 -m venv .venv) 2>/dev/null || python3 -m venv .venv
  fi
fi

pip_install() {
  if [ "$USE_UV" = "1" ]; then
    uv pip install --python "$MOSS/.venv/bin/python" "$@"
  else
    "$MOSS/.venv/bin/pip" install "$@"
  fi
}

# 3. Pick a CUDA wheel index matching the installed driver.
CUDA_TAG="cu124"
if command -v nvidia-smi >/dev/null 2>&1; then
  DRV_MAJOR="$(nvidia-smi | grep -oP 'CUDA Version:\s*\K[0-9]+' | head -1)"
  case "$DRV_MAJOR" in
    13) CUDA_TAG="cu130" ;;
    12) CUDA_TAG="cu128" ;;
  esac
fi
echo "==> Ensuring PyTorch ($CUDA_TAG)…"
pip_install "torch>=2.13,<2.15" torchvision --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

# 4. KohakuMossTrainer itself, editable from this checkout (NOT the PyPI
#    wheel): as of the 0.1.0 PyPI release, the wheel is missing its packaged
#    Qwen3.5 tokenizer assets (kohakumoss/encoders/text/assets/), which breaks
#    text-encoding with "need sentencepiece or tiktoken" / tokenizer errors
#    even though the encoder loads fine. The git checkout has those files, and
#    an editable install uses them straight from source.
echo "==> Installing KohakuMossTrainer (editable, from source)…"
pip_install -e ".[cuda]" --extra-index-url "https://download.pytorch.org/whl/$CUDA_TAG"
pip_install sentencepiece

echo "==> Done. KohakuMossTrainer is ready. Download the MOSS-Image weights next (Setup tab)."
