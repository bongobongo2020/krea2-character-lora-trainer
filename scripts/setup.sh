#!/bin/bash
# Optional all-in-one CLI setup. The web UI's Setup tab does the same with
# buttons. By default this installs the recommended Turbo engine (AI Toolkit).
#
#   bash scripts/setup.sh              # install AI Toolkit (Turbo)
#   bash scripts/setup.sh musubi       # also install musubi-tuner (Raw)
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Installing AI Toolkit (Krea 2 Turbo engine)…"
bash "$HERE/install_aitoolkit.sh"

if [ "$1" = "musubi" ] || [ "$1" = "all" ]; then
  echo "==> Installing Musubi-Tuner (Krea 2 Raw engine)…"
  bash "$HERE/install_musubi.sh"
  echo "==> For the Raw engine, place krea2_raw.safetensors + qwen_image_vae.safetensors in ./models"
fi

echo "==> Setup complete. Start the app with:  bash run.sh"
