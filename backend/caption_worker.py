"""Standalone Qwen-VL auto-captioner.

Run inside an engine venv that has torch + transformers (it is launched as a
subprocess by the web backend, so the web server stays lightweight). Writes a
``.txt`` caption next to every image in a directory.

Usage:
    python caption_worker.py <images_dir> <trigger_word> <model_id> <overwrite:0|1>

Prints one progress line per image (so the UI can stream it).
"""
from __future__ import annotations

import sys
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

PROMPT = (
    "Write a single concise training caption for this image as a comma-separated "
    "list of visual tags. Cover the main subject, appearance, hair, clothing, "
    "pose/expression, and background/setting. Output only the tags, no preamble, "
    "no sentences, do not start with 'a' or 'the'."
)


def main() -> int:
    images_dir = Path(sys.argv[1])
    trigger = sys.argv[2] if len(sys.argv) > 2 else ""
    model_id = sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3-VL-4B-Instruct"
    overwrite = len(sys.argv) > 4 and sys.argv[4] == "1"

    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    images = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    print(f"Found {len(images)} image(s) in {images_dir}", flush=True)
    if not images:
        print("Nothing to caption.", flush=True)
        return 0

    cuda = torch.cuda.is_available()
    if cuda:
        free, total = torch.cuda.mem_get_info()
        print(f"GPU free: {free/1e9:.1f} GB / {total/1e9:.1f} GB", flush=True)

    # Load strategy, in order of preference:
    #   1) GPU 4-bit (bitsandbytes) — fits a VLM in a few GB, survives a busy card
    #   2) GPU bf16 — if 4-bit unavailable but plenty of VRAM is free
    #   3) CPU float32 — always works (slow), uses system RAM not VRAM
    model = None
    device = "cpu"
    processor = AutoProcessor.from_pretrained(model_id)

    if cuda:
        try:
            from transformers import BitsAndBytesConfig
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            print(f"Loading {model_id} in 4-bit on GPU…", flush=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, quantization_config=qcfg, device_map="cuda",
                low_cpu_mem_usage=True)
            device = "cuda"
        except Exception as e:
            print(f"4-bit GPU load failed ({e}); falling back to CPU.", flush=True)
            torch.cuda.empty_cache()

    if model is None:  # CPU fallback (slow but reliable; you have plenty of RAM)
        print(f"Loading {model_id} on CPU (float32)…", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float32, low_cpu_mem_usage=True)
        device = "cpu"

    model.eval()
    print(f"Model loaded on {device}. Captioning…", flush=True)

    # Cap vision tokens: large training images blow up VRAM, so downscale the
    # long side. 1024px is plenty for a descriptive caption.
    MAX_SIDE = 1024

    trigger_norm = trigger.strip().strip(",").strip().lower()

    def is_placeholder(text: str) -> bool:
        # A caption that is empty or just the seeded trigger word isn't a real
        # caption — (re)generate it even when not overwriting.
        t = text.strip().strip(",").strip().lower()
        return t == "" or t == trigger_norm

    done = 0
    for i, p in enumerate(images, 1):
        cap_file = p.with_suffix(".txt")
        existing = cap_file.read_text() if cap_file.exists() else ""
        if not overwrite and not is_placeholder(existing):
            print(f"[{i}/{len(images)}] {p.name}: (kept existing caption)", flush=True)
            continue
        try:
            image = Image.open(p).convert("RGB")
            if max(image.size) > MAX_SIDE:
                image.thumbnail((MAX_SIDE, MAX_SIDE))
            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT}]}]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            trimmed = generated[:, inputs.input_ids.shape[1]:]
            caption = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            caption = " ".join(caption.split()).strip().strip(".")
            if trigger and trigger.lower() not in caption.lower():
                caption = f"{trigger}, {caption}"
            cap_file.write_text(caption)
            done += 1
            print(f"[{i}/{len(images)}] {p.name}: {caption}", flush=True)
            del inputs, generated, trimmed
        except Exception as e:  # keep going on a single bad image
            print(f"[{i}/{len(images)}] {p.name}: ERROR {e}", flush=True)
        finally:
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"Done. Wrote {done} caption(s).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
