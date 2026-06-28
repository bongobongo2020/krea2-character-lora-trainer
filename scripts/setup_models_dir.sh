#!/bin/bash
# Centralize all model downloads under /mnt/ai-models so they are shared and
# never re-downloaded. Because /mnt is the same filesystem as your home dir,
# moving the existing 72 GB Hugging Face cache is an instant rename (no copy).
#
# Run once (needs sudo for the /mnt mkdir):
#     bash scripts/setup_models_dir.sh
set -e

DEST="${MODELS_DIR:-/mnt/ai-models}"
HF_SRC="$HOME/.cache/huggingface"
HF_DEST="$DEST/huggingface"

echo "==> Centralizing models in: $DEST"
sudo mkdir -p "$DEST"
sudo chown "$(id -un):$(id -gn)" "$DEST"

if [ -L "$HF_SRC" ]; then
  echo "==> $HF_SRC is already a symlink — nothing to move."
elif [ -d "$HF_SRC" ]; then
  if [ -e "$HF_DEST" ]; then
    echo "==> $HF_DEST already exists; leaving the existing cache in place."
  else
    echo "==> Moving existing HF cache -> $HF_DEST (instant, same filesystem)…"
    mv "$HF_SRC" "$HF_DEST"
  fi
  ln -s "$HF_DEST" "$HF_SRC"
  echo "==> Symlinked $HF_SRC -> $HF_DEST"
else
  mkdir -p "$HF_DEST"
  echo "==> Created empty cache at $HF_DEST"
fi

echo "==> Done. The app's default hf_home ($HF_DEST) now points here."
echo "    Verify in the Setup tab — the model-cache list should show your"
echo "    Krea-2-Turbo / adapter / Qwen-VL models as 'cached'."
