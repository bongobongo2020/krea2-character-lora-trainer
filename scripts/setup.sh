#!/bin/bash
# Optional all-in-one CLI setup. The web UI's Setup tab does the same with
# buttons. By default this installs the recommended Turbo engine (AI Toolkit).
#
#   bash scripts/setup.sh              # install AI Toolkit (Turbo)
#   bash scripts/setup.sh musubi       # also install musubi-tuner (Raw)
#   bash scripts/setup.sh moss         # also install KohakuMossTrainer (MOSS-Image)
#   bash scripts/setup.sh all          # install every engine
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==> Installing AI Toolkit (Krea 2 Turbo engine)…"
bash "$HERE/install_aitoolkit.sh"

if [ "$1" = "musubi" ] || [ "$1" = "all" ]; then
  echo "==> Installing Musubi-Tuner (Krea 2 Raw engine)…"
  bash "$HERE/install_musubi.sh"
  echo "==> For the Raw engine, place krea2_raw.safetensors + qwen_image_vae.safetensors in ./models"
fi

if [ "$1" = "moss" ] || [ "$1" = "all" ]; then
  echo "==> Installing KohakuMossTrainer (MOSS-Image engine)…"
  bash "$HERE/install_moss.sh"
  echo "==> For the MOSS engine, download the weights from the Setup tab, or:"
  echo "    hf download Comfy-Org/MOSS-Image diffusion_models/moss_image_base_int8_convrot.safetensors \\"
  echo "        text_encoders/qwen3.5_9b_int8_convrot.safetensors vae/flux2-vae_small.safetensors \\"
  echo "        --local-dir ./models/moss-image"
fi

echo "==> Setup complete. Start the app with:  bash run.sh"
