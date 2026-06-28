"""Project storage + training process management (two engines).

A *project* is one character LoRA. Depending on its ``engine`` it is turned into
either:

* **aitoolkit** — an AI Toolkit YAML config run with ``python run.py config.yaml``,
  training on ``krea/Krea-2-Turbo`` with the de-distill assistant LoRA. Produces
  a Turbo-native LoRA.
* **musubi** — a ``dataset.toml`` + ``accelerate launch krea2_train_network.py``
  command training on the Krea 2 raw model.

Only one training run executes at a time (single-GPU assumption).
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from . import config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

ENGINES = ("aitoolkit", "musubi")
DEFAULT_ENGINE = "aitoolkit"

# AI Toolkit (Krea 2 Turbo) defaults.
DEFAULT_PARAMS_AITK: dict[str, Any] = {
    "trigger_word": "mychar",
    "resolution": 1024,
    "network_dim": 32,
    "network_alpha": 32,
    "learning_rate": 1e-4,
    "steps": 2000,
    "save_every": 250,
    "sample_every": 250,
    "batch_size": 1,
    "quantize": True,
    "low_vram": True,
    "seed": 42,
}

# musubi-tuner (Krea 2 Raw) defaults, matching the reference 12 GB recipe.
DEFAULT_PARAMS_MUSUBI: dict[str, Any] = {
    "trigger_word": "mychar",
    "resolution": 1024,
    "num_repeats": 10,
    "network_dim": 32,
    "network_alpha": 32,
    "learning_rate": 1e-4,
    "max_train_epochs": 10,
    "save_every_n_epochs": 2,
    "blocks_to_swap": 26,
    "fp8": True,
    "discrete_flow_shift": 2.5,
    "seed": 42,
}


def default_params(engine: str) -> dict[str, Any]:
    return dict(DEFAULT_PARAMS_AITK if engine == "aitoolkit" else DEFAULT_PARAMS_MUSUBI)


def recommend_params(engine: str, vram_mb: int | None) -> dict[str, Any]:
    """VRAM-based parameter tuning so training fits the card without OOM.

    Returns only the params that should override the engine defaults; the UI
    shows these and the user can still adjust them.
    """
    if not vram_mb:
        return {}
    gb = vram_mb / 1024
    if engine == "aitoolkit":  # Krea 2 Turbo, 12B
        if gb >= 40:
            return {"quantize": False, "low_vram": False, "resolution": 1024}
        if gb >= 22:
            return {"quantize": True, "low_vram": True, "resolution": 1024}
        if gb >= 14:
            return {"quantize": True, "low_vram": True, "resolution": 768}
        return {"quantize": True, "low_vram": True, "resolution": 512}
    # musubi raw, 12B — trade VRAM for speed via block swap
    if gb >= 32:
        return {"blocks_to_swap": 0, "fp8": True, "resolution": 1024}
    if gb >= 22:
        return {"blocks_to_swap": 16, "fp8": True, "resolution": 1024}
    if gb >= 14:
        return {"blocks_to_swap": 20, "fp8": True, "resolution": 1024}
    if gb >= 11:
        return {"blocks_to_swap": 26, "fp8": True, "resolution": 1024}
    return {"blocks_to_swap": 36, "fp8": True, "resolution": 768}


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return s.lower() or "project"


@dataclass
class Project:
    id: str
    name: str
    engine: str = DEFAULT_ENGINE
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS_AITK))

    # --- paths -----------------------------------------------------------
    @property
    def dir(self) -> Path:
        return config.projects_dir() / self.id

    @property
    def images_dir(self) -> Path:
        return self.dir / "dataset" / "images"

    @property
    def cache_dir(self) -> Path:
        return self.dir / "dataset" / "cache"

    @property
    def output_dir(self) -> Path:
        return self.dir / "output"

    @property
    def logs_dir(self) -> Path:
        return self.dir / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "train.log"

    @property
    def toml_file(self) -> Path:
        return self.dir / "dataset.toml"

    @property
    def yaml_file(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def meta_file(self) -> Path:
        return self.dir / "project.json"

    def ensure_dirs(self) -> None:
        for d in (self.images_dir, self.cache_dir, self.output_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.ensure_dirs()
        self.meta_file.write_text(json.dumps(
            {"id": self.id, "name": self.name, "engine": self.engine, "params": self.params},
            indent=2))

    @classmethod
    def load(cls, project_id: str) -> Optional["Project"]:
        meta = config.projects_dir() / project_id / "project.json"
        if not meta.exists():
            return None
        data = json.loads(meta.read_text())
        engine = data.get("engine", DEFAULT_ENGINE)
        params = default_params(engine)
        params.update(data.get("params", {}))
        return cls(id=data["id"], name=data["name"], engine=engine, params=params)

    def output_name(self) -> str:
        return f"{_slug(self.name)}_krea2"

    # --- dataset helpers -------------------------------------------------
    def list_images(self) -> list[dict[str, Any]]:
        items = []
        for p in sorted(self.images_dir.iterdir()) if self.images_dir.exists() else []:
            if p.suffix.lower() in IMAGE_EXTS:
                caption_file = p.with_suffix(".txt")
                caption = caption_file.read_text().strip() if caption_file.exists() else ""
                items.append({"filename": p.name, "caption": caption})
        return items

    def _unique_image_path(self, stem: str, ext: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem) or "img"
        dest = self.images_dir / f"{safe}{ext}"
        i = 1
        while dest.exists():
            dest = self.images_dir / f"{safe}_{i}{ext}"
            i += 1
        return dest

    def import_zip(self, data: bytes, default_caption: str = "") -> list[str]:
        import io
        import zipfile

        saved: list[str] = []
        stem_to_image: dict[str, Path] = {}
        pending_captions: dict[str, str] = {}
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise ValueError("That file is not a valid .zip archive.")

        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            stem, ext = Path(name).stem, Path(name).suffix.lower()
            if not stem or name.startswith("."):
                continue
            if ext in IMAGE_EXTS:
                dest = self._unique_image_path(stem, ext)
                dest.write_bytes(zf.read(info))
                stem_to_image[stem] = dest
                saved.append(dest.name)
            elif ext == ".txt":
                pending_captions[stem] = zf.read(info).decode("utf-8", errors="replace").strip()

        for stem, img_path in stem_to_image.items():
            caption_file = img_path.with_suffix(".txt")
            if stem in pending_captions:
                caption_file.write_text(pending_captions[stem])
            elif not caption_file.exists():
                caption_file.write_text(default_caption)
        return saved

    # --- config generation ----------------------------------------------
    def _sample_prompts(self) -> list[str]:
        t = self.params.get("trigger_word", "").strip() or "mychar"
        return [
            f"{t}, portrait, studio lighting, white background",
            f"{t}, full body, standing in a park, sunny day",
            f"{t}, close-up of the face, detailed, soft light",
            f"{t}, in a city street at night, neon lights",
        ]

    def generate_aitoolkit_yaml(self, settings: dict[str, Any]) -> str:
        p = self.params
        res = int(p["resolution"])
        model: dict[str, Any] = {
            "name_or_path": settings["base_model"],
            "arch": "krea2",
            "assistant_lora_path": settings["assistant_lora"],
            "quantize": bool(p.get("quantize", True)),
            "low_vram": bool(p.get("low_vram", True)),
        }
        if p.get("quantize", True):
            model["qtype"] = "qfloat8"
            model["quantize_te"] = True
            model["qtype_te"] = "qfloat8"

        cfg = {
            "job": "extension",
            "config": {
                "name": self.output_name(),
                "process": [{
                    "type": "sd_trainer",
                    "training_folder": str(self.output_dir),
                    "device": "cuda:0",
                    "network": {"type": "lora",
                                "linear": int(p["network_dim"]),
                                "linear_alpha": int(p["network_alpha"])},
                    "save": {"dtype": "float16",
                             "save_every": int(p["save_every"]),
                             "max_step_saves_to_keep": 6},
                    "datasets": [{
                        "folder_path": str(self.images_dir),
                        "caption_ext": "txt",
                        "caption_dropout_rate": 0.05,
                        "cache_latents_to_disk": True,
                        "resolution": [res],
                    }],
                    "train": {
                        "batch_size": int(p["batch_size"]),
                        "steps": int(p["steps"]),
                        "gradient_accumulation": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        # Cache text embeddings + latents to disk so the text
                        # encoder and VAE can be freed from VRAM during training
                        # (big saving on 24 GB cards).
                        "cache_text_embeddings": True,
                        # Skip the step-0 baseline sample: on Windows that first
                        # 1024px inference can spill into shared VRAM and stall
                        # the run for an hour before training begins.
                        "skip_first_sample": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": "adamw8bit",
                        "lr": float(p["learning_rate"]),
                        "dtype": "bf16",
                    },
                    "model": model,
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": int(p["sample_every"]),
                        # 512px keeps periodic sampling cheap so it doesn't
                        # tip a low-VRAM card into shared-memory crawl.
                        "width": 512,
                        "height": 512,
                        "prompts": self._sample_prompts(),
                        "neg": "",
                        "seed": int(p["seed"]),
                        "walk_seed": True,
                        # Turbo inference settings.
                        "guidance_scale": 1,
                        "sample_steps": 8,
                    },
                }],
            },
            "meta": {"name": self.output_name(), "version": "1.0"},
        }
        text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
        self.yaml_file.write_text(text)
        return text

    def generate_musubi_toml(self) -> str:
        p = self.params
        res = int(p["resolution"])
        toml = (
            "[general]\n"
            f"resolution = [{res}, {res}]\n"
            'caption_extension = ".txt"\n'
            "batch_size = 1\n"
            "enable_bucket = true\n"
            "bucket_no_upscale = false\n\n"
            "[[datasets]]\n"
            f'image_directory = "{self.images_dir}"\n'
            f'cache_directory = "{self.cache_dir}"\n'
            f"num_repeats = {int(p['num_repeats'])}\n"
        )
        self.toml_file.write_text(toml)
        return toml

    def build_command(self, settings: dict[str, Any]) -> tuple[list[str], str, dict[str, str]]:
        """Return (argv, cwd, extra_env) for the project's engine."""
        if self.engine == "aitoolkit":
            self.generate_aitoolkit_yaml(settings)
            argv = [settings["aitoolkit_python"], "run.py", str(self.yaml_file)]
            # expandable_segments prevents the fragmentation OOM that hits 24 GB
            # cards partway through training. HF_HOME keeps all caches in one place.
            env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                   **config.hf_env(settings)}
            if settings.get("hf_token"):
                env["HF_TOKEN"] = settings["hf_token"]
                env["HUGGING_FACE_HUB_TOKEN"] = settings["hf_token"]
            return argv, settings["aitoolkit_dir"], env

        # musubi
        p = self.params
        self.generate_musubi_toml()
        argv = [
            settings["accelerate_bin"], "launch",
            "--num_cpu_threads_per_process", "1", "--mixed_precision", "bf16",
            str(Path(settings["musubi_dir"]) / "src" / "musubi_tuner" / "krea2_train_network.py"),
            "--dit", settings["dit_model"], "--vae", settings["vae_model"],
            "--dataset_config", str(self.toml_file),
            "--sdpa", "--mixed_precision", "bf16",
        ]
        if p.get("fp8", True):
            argv += ["--fp8_base", "--fp8_scaled"]
        argv += [
            "--blocks_to_swap", str(int(p["blocks_to_swap"])),
            "--timestep_sampling", "shift", "--weighting_scheme", "none",
            "--discrete_flow_shift", str(float(p["discrete_flow_shift"])),
            "--network_module", "networks.lora_krea2",
            "--network_dim", str(int(p["network_dim"])),
            "--network_alpha", str(int(p["network_alpha"])),
            "--optimizer_type", "adamw",
            "--learning_rate", str(float(p["learning_rate"])),
            "--max_train_epochs", str(int(p["max_train_epochs"])),
            "--save_every_n_epochs", str(int(p["save_every_n_epochs"])),
            "--gradient_checkpointing", "--seed", str(int(p["seed"])),
            "--output_dir", str(self.output_dir),
            "--output_name", self.output_name(),
            "--logging_dir", str(self.logs_dir),
        ]
        env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
               **config.hf_env(settings)}
        return argv, settings["musubi_dir"], env

    def list_outputs(self) -> list[dict[str, Any]]:
        items = []
        if self.output_dir.exists():
            for p in self.output_dir.rglob("*.safetensors"):
                # Intermediate checkpoints carry a step/epoch suffix
                # (e.g. ..._000001750.safetensors); the final one does not.
                m = re.search(r"[-_](\d{2,})\.safetensors$", p.name)
                items.append({
                    "filename": p.name,
                    "relpath": str(p.relative_to(self.output_dir)),
                    "size": p.stat().st_size,
                    "step": int(m.group(1)) if m else None,
                })
        # Numbered checkpoints in step order, the final (unnumbered) one last.
        items.sort(key=lambda x: (x["step"] is None, x["step"] or 0))
        total = int(self.params.get("steps") or 0)
        for it in items:
            if it["step"] is None:
                it["label"] = f"Final{f' — {total} steps' if total else ''}"
                it["final"] = True
            else:
                it["label"] = f"Step {it['step']}"
                it["final"] = False
        return items

    def list_samples(self) -> list[dict[str, Any]]:
        """Preview images written during training, grouped by step.

        AI Toolkit names them ``<timestamp>__<step>_<promptindex>.jpg``.
        """
        by_step: dict[int, list[dict[str, Any]]] = {}
        if self.output_dir.exists():
            for sdir in self.output_dir.rglob("samples"):
                if not sdir.is_dir():
                    continue
                for img in sorted(sdir.iterdir()):
                    if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                        continue
                    m = re.search(r"__(\d+)_(\d+)\.", img.name)
                    step = int(m.group(1)) if m else 0
                    idx = int(m.group(2)) if m else 0
                    by_step.setdefault(step, []).append({
                        "relpath": str(img.relative_to(self.output_dir)),
                        "index": idx,
                    })
        groups = []
        for step in sorted(by_step):
            imgs = sorted(by_step[step], key=lambda x: x["index"])
            groups.append({"step": step, "label": "Before training" if step == 0 else f"Step {step}",
                           "images": imgs})
        return groups

    def find_output(self, relpath: str) -> Optional[Path]:
        # Guard against path traversal; only allow files under output_dir.
        target = (self.output_dir / relpath).resolve()
        if self.output_dir.resolve() in target.parents and target.exists():
            return target
        return None


class TrainingManager:
    """Supervises the single active training subprocess."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._project_id: Optional[str] = None
        self._lock = threading.Lock()
        self._started_at: float = 0.0
        self._command: str = ""

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def active_project(self) -> Optional[str]:
        return self._project_id if self.is_running() else None

    def validate(self, project: Project, settings: dict[str, Any]) -> list[str]:
        """Public preflight check — returns a list of problems (empty if OK)."""
        return self._validate(project, settings)

    def _validate(self, project: Project, settings: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        images = project.list_images()
        if not images:
            problems.append("No training images uploaded.")
        if any(not img["caption"] for img in images):
            problems.append("Some images have no caption.")

        env = config.check_environment(settings)["engines"][project.engine]
        for item in env["items"]:
            if not item["exists"]:
                problems.append(f"Missing: {item['label']} ({item['path']})")
        return problems

    def start(self, project: Project, settings: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self.is_running():
                raise RuntimeError(
                    f"A training run is already in progress (project {self._project_id}).")

            problems = self._validate(project, settings)
            if problems:
                raise RuntimeError("; ".join(problems))

            project.ensure_dirs()
            argv, cwd, extra_env = project.build_command(settings)
            self._command = " ".join(shlex.quote(c) for c in argv)

            env = {**os.environ, **extra_env, "PYTHONUNBUFFERED": "1"}

            log = project.log_file
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as fh:
                fh.write(f"# Engine: {project.engine}\n# Command:\n{self._command}\n\n")
                fh.flush()
                self._proc = subprocess.Popen(
                    argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=env, bufsize=1, text=True)
            self._project_id = project.id
            self._started_at = time.time()

            threading.Thread(target=self._pump_logs, args=(self._proc, log),
                             daemon=True).start()
            return {"command": self._command, "project_id": project.id}

    def _pump_logs(self, proc: subprocess.Popen, log: Path) -> None:
        assert proc.stdout is not None
        with log.open("a") as fh:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
        proc.wait()
        with log.open("a") as fh:
            fh.write(f"\n# Process exited with code {proc.returncode}\n")

    def stop(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                # SIGINT lets the trainer checkpoint+exit cleanly on POSIX; on
                # Windows it isn't deliverable, so terminate directly.
                try:
                    if os.name == "nt":
                        self._proc.terminate()
                    else:
                        self._proc.send_signal(signal.SIGINT)
                except Exception:
                    self._proc.terminate()
                try:
                    self._proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self._proc.kill()

    def status(self, project: Project) -> dict[str, Any]:
        running = self.is_running() and self._project_id == project.id
        progress = _parse_progress(project.log_file)
        exit_code = None
        if self._project_id == project.id and self._proc is not None and not self.is_running():
            exit_code = self._proc.returncode
        return {
            "running": running,
            "started_at": self._started_at if running else None,
            "elapsed": (time.time() - self._started_at) if running else None,
            "exit_code": exit_code,
            "command": self._command if self._project_id == project.id else "",
            **progress,
        }


_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)")
_LOSS_RE = re.compile(r"loss[=:\s]+([0-9]*\.?[0-9]+)", re.IGNORECASE)
_EPOCH_RE = re.compile(r"epoch[:\s]+(\d+)\s*/\s*(\d+)", re.IGNORECASE)


def _parse_progress(log_file: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"step": None, "total_steps": None, "loss": None,
                              "epoch": None, "total_epochs": None}
    if not log_file.exists():
        return result
    try:
        tail = _read_tail(log_file, 8000)
    except OSError:
        return result
    for line in reversed(tail.splitlines()):
        if "steps:" in line.lower() or "/it" in line or "it/s" in line or "%|" in line:
            m = _STEP_RE.search(line)
            if m and result["step"] is None:
                result["step"], result["total_steps"] = int(m.group(1)), int(m.group(2))
            lm = _LOSS_RE.search(line)
            if lm and result["loss"] is None:
                result["loss"] = float(lm.group(1))
        em = _EPOCH_RE.search(line)
        if em and result["epoch"] is None:
            result["epoch"], result["total_epochs"] = int(em.group(1)), int(em.group(2))
        if result["step"] is not None and result["loss"] is not None:
            break
    return result


def _read_tail(path: Path, nbytes: int) -> str:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - nbytes))
        return fh.read().decode("utf-8", errors="replace")


# --- project CRUD --------------------------------------------------------

def create_project(name: str, engine: str = DEFAULT_ENGINE) -> Project:
    if engine not in ENGINES:
        engine = DEFAULT_ENGINE
    pid = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
    params = default_params(engine)
    # Auto-tune for the detected GPU so a new project fits the card by default.
    hw = config.detect_hardware()
    params.update(recommend_params(engine, hw.get("vram_mb")))
    proj = Project(id=pid, name=name, engine=engine, params=params)
    proj.save()
    return proj


def list_projects() -> list[dict[str, Any]]:
    out = []
    pdir = config.projects_dir()
    if pdir.exists():
        for d in sorted(pdir.iterdir()):
            proj = Project.load(d.name)
            if proj:
                out.append({
                    "id": proj.id, "name": proj.name, "engine": proj.engine,
                    "images": len(proj.list_images()), "outputs": len(proj.list_outputs()),
                })
    return out


manager = TrainingManager()
