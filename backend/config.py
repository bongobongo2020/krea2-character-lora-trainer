"""Application configuration and persistent settings.

Two training engines are supported, each with its own paths/assets:

* **aitoolkit** — ostris/ai-toolkit training directly on the (un-gated)
  ``krea/Krea-2-Turbo`` model using the de-distill *assistant LoRA*
  (``ostris/krea2_turbo_training_adapter``). Turbo-native, recommended.
* **musubi** — kohya-ss/musubi-tuner training on the Krea 2 *raw* model; the
  resulting LoRA also transfers to Turbo.

Heavy paths are editable from the UI so the frontend can validate them before a
run and guide setup.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.environ.get("KREA_WORKSPACE", APP_ROOT / "workspace"))
SETTINGS_FILE = WORKSPACE / "settings.json"

IS_WINDOWS = os.name == "nt"


def venv_python(venv_dir: Path | str) -> str:
    """Path to the Python interpreter inside a venv (cross-platform)."""
    v = Path(venv_dir)
    return str(v / ("Scripts/python.exe" if IS_WINDOWS else "bin/python"))


def venv_bin(venv_dir: Path | str, name: str) -> str:
    """Path to a console script inside a venv (cross-platform)."""
    v = Path(venv_dir)
    return str(v / "Scripts" / f"{name}.exe") if IS_WINDOWS else str(v / "bin" / name)


def _default_models_home() -> str:
    """Default shared model cache. The Windows installer overrides this with the
    folder the user picks; on Linux we default to /mnt/ai-models."""
    if IS_WINDOWS:
        return str(Path.home() / "ai-models" / "huggingface")
    return "/mnt/ai-models/huggingface"

DEFAULT_SETTINGS: dict[str, Any] = {
    # --- AI Toolkit (Turbo) ---
    "aitoolkit_dir": str(APP_ROOT / "ai-toolkit"),
    "aitoolkit_python": venv_python(APP_ROOT / "ai-toolkit" / "venv"),
    # Base model + de-distill adapter. May be HF repo ids (auto-downloaded by
    # AI Toolkit) or local paths.
    "base_model": "krea/Krea-2-Turbo",
    # AI Toolkit expects the adapter as a 3-part hub path: org/repo/filename.
    "assistant_lora": "ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors",
    # Optional Hugging Face token (helps with rate limits; not required —
    # Krea-2-Turbo is not gated).
    "hf_token": "",
    # Central model/cache location. All engines + the captioner use this as
    # HF_HOME so models download once and are shared (no wasteful redownloads).
    # Falls back to the standard ~/.cache/huggingface if this isn't usable yet.
    "hf_home": _default_models_home(),

    # --- Auto-captioning (Qwen-VL) ---
    # Smallest/fastest Qwen-VL that does the job; defaults to one already cached.
    "caption_model": "Qwen/Qwen3-VL-4B-Instruct",
    # Python to run the captioner with (must have torch+transformers). Empty =
    # auto-pick an installed engine venv.
    "caption_python": "",
    # Automatically free VRAM (gentle ComfyUI /free unload) before training or
    # captioning when the card is too full. Never kills processes.
    "auto_free_vram": "true",

    # --- musubi-tuner (Raw) ---
    "musubi_dir": str(APP_ROOT / "musubi-tuner"),
    "accelerate_bin": venv_bin(APP_ROOT / "musubi-tuner" / ".venv", "accelerate"),
    "dit_model": str(APP_ROOT / "models" / "krea2_raw.safetensors"),
    "vae_model": str(APP_ROOT / "models" / "qwen_image_vae.safetensors"),
}

SETTINGS_KEYS = list(DEFAULT_SETTINGS.keys())


def _ensure_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "projects").mkdir(parents=True, exist_ok=True)


def load_settings() -> dict[str, Any]:
    _ensure_workspace()
    data = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            data.update(json.loads(SETTINGS_FILE.read_text()))
        except (json.JSONDecodeError, OSError):
            pass
    return data


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    _ensure_workspace()
    current = load_settings()
    for key in SETTINGS_KEYS:
        if key in settings and settings[key] is not None:
            current[key] = str(settings[key])
    SETTINGS_FILE.write_text(json.dumps(current, indent=2))
    return current


def projects_dir() -> Path:
    _ensure_workspace()
    return WORKSPACE / "projects"


def _exists(path: str) -> bool:
    return Path(path).exists()


def effective_hf_home(settings: dict[str, Any]) -> str:
    """Resolve the HF cache dir to actually use.

    Prefers the configured ``hf_home`` if it (or its parent) exists and is
    writable; otherwise falls back to the standard ~/.cache/huggingface so
    nothing breaks before the user has set up /mnt/ai-models.
    """
    home = (settings.get("hf_home") or "").strip()
    if home:
        p = Path(home)
        if p.exists() and os.access(p, os.W_OK):
            return home
        if p.parent.exists() and os.access(p.parent, os.W_OK):
            return home
    return str(Path.home() / ".cache" / "huggingface")


def hf_env(settings: dict[str, Any]) -> dict[str, str]:
    """Env vars to point a subprocess at the shared HF cache."""
    return {"HF_HOME": effective_hf_home(settings)}


def _repo_cache_dir(hub: Path, repo_id: str) -> Path:
    return hub / ("models--" + repo_id.replace("/", "--"))


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        # Skip symlinks: the HF cache snapshots are symlinks into blobs/, so
        # following them would double-count every file.
        if f.is_symlink() or not f.is_file():
            continue
        try:
            total += f.stat().st_size
        except OSError:
            pass
    return total


_HW_CACHE: dict[str, Any] | None = None


def _total_ram_mb() -> int | None:
    # Linux / macOS
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    # Windows
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return int(stat.ullTotalPhys // (1024 * 1024))
    except Exception:
        return None


def detect_hardware(refresh: bool = False) -> dict[str, Any]:
    """Detect GPU name + total VRAM and total system RAM (cross-platform)."""
    global _HW_CACHE
    if _HW_CACHE is not None and not refresh:
        return _HW_CACHE
    import shutil
    import subprocess
    gpu_name, vram_mb = None, None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
            line = out.stdout.strip().splitlines()[0]
            name, total = [x.strip() for x in line.split(",")]
            gpu_name, vram_mb = name, int(total)
        except Exception:
            pass
    _HW_CACHE = {"gpu_name": gpu_name, "vram_mb": vram_mb, "ram_mb": _total_ram_mb()}
    return _HW_CACHE


def scan_models(settings: dict[str, Any]) -> dict[str, Any]:
    """Report which models are already cached, to avoid wasteful redownloads."""
    hub = Path(effective_hf_home(settings)) / "hub"
    wanted = [
        ("base_model", settings["base_model"], "Krea-2-Turbo base"),
        ("assistant_lora", settings["assistant_lora"], "De-distill adapter"),
        ("caption_model", settings["caption_model"], "Qwen-VL captioner"),
    ]
    items = []
    for key, repo, label in wanted:
        repo_id = _hf_repo_id(repo)
        if repo_id is None:  # a local path
            items.append({"key": key, "repo": repo, "label": label,
                          "cached": _exists(repo), "size": None})
            continue
        d = _repo_cache_dir(hub, repo_id)
        items.append({"key": key, "repo": repo_id, "label": label,
                      "cached": d.exists(), "size": _dir_size(d) if d.exists() else None})
    return {"hf_home": str(hub.parent), "hub": str(hub), "models": items}


def _hf_repo_id(ref: str) -> str | None:
    """Extract a Hugging Face repo id from a model reference, or None if it's a
    local path. Refs may be ``org/name`` or ``org/name/file.safetensors``."""
    if not ref or os.path.isabs(ref) or Path(ref).exists():
        return None
    parts = ref.split("/")
    if len(parts) >= 2:
        return "/".join(parts[:2])
    return None


def _modelref_ok(ref: str) -> bool:
    """A model reference is OK if it's an existing local file, or a HF repo id.

    Repo ids (``org/name`` or ``org/name/file``) are auto-downloaded at train
    time, so we treat them as satisfied for the readiness check.
    """
    if not ref:
        return False
    if os.path.isabs(ref):
        return Path(ref).exists()
    if Path(ref).exists():  # relative local path
        return True
    return _hf_repo_id(ref) is not None  # a hub path


def check_environment(settings: dict[str, Any]) -> dict[str, Any]:
    """Per-engine readiness, consumed by the Setup tab."""
    aitk_run = Path(settings["aitoolkit_dir"]) / "run.py"
    aitk_items = [
        {"key": "aitoolkit_dir", "label": "AI Toolkit folder",
         "path": settings["aitoolkit_dir"], "exists": _exists(settings["aitoolkit_dir"])},
        {"key": "aitoolkit_run", "label": "AI Toolkit run.py",
         "path": str(aitk_run), "exists": aitk_run.exists()},
        {"key": "aitoolkit_python", "label": "AI Toolkit Python",
         "path": settings["aitoolkit_python"], "exists": _exists(settings["aitoolkit_python"])},
        {"key": "base_model", "label": "Krea-2-Turbo base model",
         "path": settings["base_model"], "exists": _modelref_ok(settings["base_model"]),
         "info": "auto-downloads"},
        {"key": "assistant_lora", "label": "De-distill adapter",
         "path": settings["assistant_lora"], "exists": _modelref_ok(settings["assistant_lora"]),
         "info": "auto-downloads"},
    ]

    musubi_script = Path(settings["musubi_dir"]) / "src" / "musubi_tuner" / "krea2_train_network.py"
    musubi_items = [
        {"key": "musubi_dir", "label": "Musubi-Tuner folder",
         "path": settings["musubi_dir"], "exists": _exists(settings["musubi_dir"])},
        {"key": "musubi_script", "label": "Krea2 training script",
         "path": str(musubi_script), "exists": musubi_script.exists()},
        {"key": "accelerate_bin", "label": "accelerate launcher",
         "path": settings["accelerate_bin"], "exists": _exists(settings["accelerate_bin"])},
        {"key": "dit_model", "label": "Krea2 raw base model",
         "path": settings["dit_model"], "exists": _exists(settings["dit_model"])},
        {"key": "vae_model", "label": "VAE model",
         "path": settings["vae_model"], "exists": _exists(settings["vae_model"])},
    ]

    return {
        "engines": {
            "aitoolkit": {
                "label": "AI Toolkit — Krea 2 Turbo (recommended)",
                "items": aitk_items,
                "ready": all(i["exists"] for i in aitk_items),
            },
            "musubi": {
                "label": "musubi-tuner — Krea 2 Raw",
                "items": musubi_items,
                "ready": all(i["exists"] for i in musubi_items),
            },
        }
    }
