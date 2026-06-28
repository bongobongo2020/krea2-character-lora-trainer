# Krea 2 Character LoRA Trainer

A simple **web UI + backend** for training character LoRAs for the **Krea 2**
image model — no command line needed. You create a project, drop in 10–20
character images, write captions with a trigger word, click **Start training**,
and download the resulting `.safetensors` LoRA.

## Two engines (pick per project)

| Engine | Trains on | Best for | Notes |
|--------|-----------|----------|-------|
| **AI Toolkit — Krea 2 Turbo** *(recommended)* | `krea/Krea-2-Turbo` + de-distill adapter | Turbo-native LoRAs | Base model is **not gated** and auto-downloads. |
| **musubi-tuner — Krea 2 Raw** | Krea 2 raw model | The original 12 GB recipe | You supply the raw model + VAE files. |

The **Turbo** engine uses [ostris/ai-toolkit](https://github.com/ostris/ai-toolkit)
with the [Krea-2-Turbo training adapter](https://huggingface.co/ostris/krea2_turbo_training_adapter):
a "de-distill" assistant LoRA that is active during training and automatically
removed at inference, so your character LoRA runs at full Turbo (8-step) speed
without breaking the distillation. The **Raw** engine reproduces the
[masafykun/krea2-character-lora](https://github.com/masafykun/krea2-character-lora)
recipe; its LoRA also transfers to Turbo.

## What you need

- A CUDA GPU. The Raw recipe fits in **12 GB** (FP8 + block swap); the Turbo
  engine wants more headroom (quantize + low-VRAM options are on by default).
- The web server itself only needs plain Python 3.9+ (`run.sh` sets it up).

## Quick start — Windows 11 (no command line)

1. Download/extract this project somewhere (e.g. `Documents\krea-lora-trainer`).
2. Open the `windows` folder and **double-click `Install.bat`**.

The installer (a guided PowerShell script) will:
- check for and, via `winget`, install missing **Python / Git / uv**;
- pop a **folder picker** to choose where models live (existing downloads there
  are reused — nothing re-downloads);
- **scan** that folder + the HF cache and report which models are already present;
- detect your **GPU VRAM and system RAM** and write tuned defaults;
- install the **AI Toolkit (Turbo) engine** and the web app;
- **optionally** offer to also install the **musubi-tuner (Raw) engine** (a
  Yes/No prompt — choose No to keep it Turbo-only);
- create a **desktop shortcut**.

Then launch with the **“Krea 2 LoRA Trainer”** desktop icon (or
`windows\Start.bat`) — your browser opens to the app automatically. Windows may
prompt to allow the app through the firewall the first time (needed only if you
want other devices on your network to reach it).

> New projects are auto-tuned to your card: the Turbo engine enables
> quantization + low-VRAM mode and picks a resolution that fits; the Raw engine
> sets `blocks_to_swap` for your VRAM. You can still adjust everything per project.

## Quick start — Linux / macOS

```bash
bash run.sh          # starts the web app
```

It binds to `0.0.0.0` so other machines on your LAN can reach it — the banner
prints the exact URL, e.g. `http://192.168.1.10:8000`. If a device can't connect,
allow the port in the host firewall: `sudo ufw allow 8000/tcp`. To restrict the
app to this machine only, run `HOST=127.0.0.1 bash run.sh`.

That's the only terminal command. Everything else is buttons in the UI:

1. Open the **Setup** tab — a readiness checklist shows green/red dots per engine.
2. Click **⬇ Install AI Toolkit** (recommended). It clones AI Toolkit and builds
   its Python + PyTorch environment, streaming a live log. Optionally click
   **Pre-download model + adapter** to cache Krea-2-Turbo ahead of time.
   *(For the Raw engine instead, expand it and click Install musubi-tuner, then
   download the raw model + VAE — paste a URL or a Hugging Face repo + filename.)*

When an engine shows **ready**, you can train with it.

## How to train a LoRA (in the UI)

1. **Projects → Create:** name it and pick the engine (Turbo is the default).
2. **Step 1 — Add images:** set a trigger word, then drag in images **or a .zip**
   (a zip may include matching `.txt` caption files — same base name as each
   image). Click **✨ Auto-caption with Qwen-VL** to caption every image
   automatically (see below), or write them by hand. Then **Save captions**.
   *Apply to all captions* prepends the trigger word everywhere.
3. **Step 2 — Settings:** sensible defaults per engine (rank/alpha 32, LR 1e-4).
   Expand the preview to see the exact config (`config.yaml` for Turbo /
   `dataset.toml` for Raw) and command that will run.
4. **Step 3 — Train:** click **Start training**. Live logs stream in and a
   progress bar tracks steps/loss. You can **Stop** at any time.
5. **Step 4 — Download:** checkpoints appear here for download as they're saved.

### Using the LoRA on Krea 2 Turbo (ComfyUI)

Load the LoRA at **strength 0.8**, set **CFG 1.0**, sampler **ER-SDE**, **8 steps**,
and include your trigger word in the prompt. (The Turbo engine produces these
natively; a Raw-engine LoRA also works with the same settings.)

## Auto-captioning (Qwen-VL)

The **✨ Auto-caption** button captions all images with a Qwen-VL model
(default `Qwen/Qwen3-VL-4B-Instruct` — the smallest one likely already in your
cache). It runs as a subprocess in an installed engine venv, streams progress
live, and writes a `.txt` next to each image (prefixed with your trigger word).
The model is loaded in **4-bit** so it fits in a few GB of VRAM even while other
apps (e.g. ComfyUI) hold the card; if the GPU can't fit it, it falls back to CPU.

## VRAM / avoiding OOM

Krea 2 is a 12B model. Training needs most of a 24 GB card, so the Train step
shows a **VRAM banner** (free GB + what's holding it). The Turbo engine is tuned
to fit 24 GB: FP8/int8 quantization of the model **and** text encoder, latent +
text-embedding caching (so the VAE/encoder leave VRAM), gradient checkpointing,
and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to avoid fragmentation
OOM.

**Automatic VRAM freeing.** When you start training or captioning, if the card
is too full the backend automatically frees VRAM by gently unloading any running
ComfyUI (via its `/free` API — models are dropped but ComfyUI keeps running). It
**never kills processes** automatically. The Train step shows a toggle (on by
default) and reports what it freed; manual **Unload / End process** buttons are
there as a fallback.

## Central model cache (no redownloads)

All engines + the captioner share one Hugging Face cache via `HF_HOME`
(Setup → Advanced → *Central model cache*, default `/mnt/ai-models/huggingface`).
To put everything under `/mnt/ai-models` without re-downloading your existing
72 GB cache, run once:

```bash
bash scripts/setup_models_dir.sh   # sudo for the /mnt mkdir; the move is instant
```

The Setup tab's **model-cache** list shows which models are already cached vs.
will download.

## How it works

```
frontend/        Vanilla HTML/CSS/JS single-page app (served by the backend)
backend/
  main.py        FastAPI: REST API + serves the frontend
  trainer.py     Engine-aware project storage, config generation, training subprocess
  setup_tasks.py UI-driven install + download jobs (streamed logs)
  config.py      Editable paths/models per engine + readiness checks
scripts/
  install_aitoolkit.sh   Run by "Install AI Toolkit" (Turbo engine)
  install_musubi.sh      Run by "Install musubi-tuner" (Raw engine)
  setup.sh               Optional all-in-one CLI setup (buttons do the same)
run.sh           Start the web app
workspace/       Created at runtime: per-project datasets, configs, logs, outputs
```

Each project is a folder under `workspace/projects/<id>/` with the uploaded
images + caption `.txt` files, the generated `config.yaml`/`dataset.toml`,
`logs/train.log`, and `output/` checkpoints.

- **Turbo:** runs `python run.py config.yaml` in AI Toolkit with `arch: krea2`,
  `name_or_path: krea/Krea-2-Turbo`, and
  `assistant_lora_path: ostris/krea2_turbo_training_adapter`.
- **Raw:** runs the reference `accelerate launch … krea2_train_network.py …`
  with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Only **one** training run executes at a time (single-GPU assumption).

## Notes

- Krea-2-Turbo is openly downloadable; the Turbo engine fetches it (and the
  adapter) automatically on first run. A Hugging Face token (Setup → Advanced)
  is optional and only helps with rate limits.
- Captioning is manual by design (small character datasets caption best by hand).
