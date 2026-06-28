"""Auto-captioning jobs (Qwen-VL), run as a streamed background subprocess."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from . import config
from .setup_tasks import JobRunner
from .trainer import Project

APP_ROOT = Path(__file__).resolve().parent.parent

# Dedicated runner/log so captioning is independent of the setup-install job.
runner = JobRunner(config.WORKSPACE / "caption.log")


def resolve_caption_python(settings: dict[str, Any]) -> str:
    """Pick a Python that has torch+transformers (an installed engine venv)."""
    explicit = (settings.get("caption_python") or "").strip()
    if explicit and Path(explicit).exists():
        return explicit
    candidates = [
        settings.get("aitoolkit_python", ""),
        config.venv_python(Path(settings.get("musubi_dir", "")) / ".venv"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return sys.executable


def start(project: Project, settings: dict[str, Any], overwrite: bool) -> dict[str, Any]:
    images = project.list_images()
    if not images:
        raise ValueError("No images to caption — upload some first.")

    py = resolve_caption_python(settings)
    model = settings.get("caption_model") or "Qwen/Qwen3-VL-4B-Instruct"
    worker = APP_ROOT / "backend" / "caption_worker.py"
    trigger = str(project.params.get("trigger_word", ""))

    env = config.hf_env(settings)
    if settings.get("hf_token"):
        env["HF_TOKEN"] = settings["hf_token"]

    runner.start(
        f"Auto-captioning {len(images)} image(s) with {model}",
        [py, str(worker), str(project.images_dir), trigger, model, "1" if overwrite else "0"],
        env=env,
    )
    return {"started": True, "model": model, "python": py}
