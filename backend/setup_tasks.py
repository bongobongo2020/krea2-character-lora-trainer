"""Background setup jobs driven from the UI (no terminal needed).

Handles the one-time heavy operations that previously required the command
line: installing Musubi-Tuner + its training venv, and downloading the Krea 2
base model and VAE. Only one setup job runs at a time and its output is streamed
to the frontend exactly like training logs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

from . import config

APP_ROOT = Path(__file__).resolve().parent.parent
JOB_LOG = config.WORKSPACE / "setup.log"


class JobRunner:
    """Runs a single background setup job and streams its output to a log."""

    def __init__(self, log_file: Path = JOB_LOG) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._name: str = ""
        self._lock = threading.Lock()
        self._started_at: float = 0.0
        self._log = log_file

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, name: str, cmd: list[str], env: Optional[dict[str, str]] = None,
              cwd: Optional[str] = None) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError(f"A task is already running: {self._name}")
            self._name = name
            self._started_at = time.time()
            config.WORKSPACE.mkdir(parents=True, exist_ok=True)
            full_env = {**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1"}
            with self._log.open("w") as fh:
                fh.write(f"# {name}\n\n")
                self._proc = subprocess.Popen(
                    cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env=full_env, bufsize=1, text=True,
                )
            threading.Thread(target=self._pump, args=(self._proc,), daemon=True).start()

    def _pump(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        with self._log.open("a") as fh:
            for line in proc.stdout:
                fh.write(line)
                fh.flush()
        proc.wait()
        with self._log.open("a") as fh:
            ok = "✓ finished successfully" if proc.returncode == 0 else f"✗ failed (exit {proc.returncode})"
            fh.write(f"\n# {ok}\n")

    def status(self) -> dict[str, Any]:
        running = self.is_running()
        exit_code = None if running else (self._proc.returncode if self._proc else None)
        return {
            "running": running,
            "name": self._name,
            "exit_code": exit_code,
            "elapsed": (time.time() - self._started_at) if running else None,
        }

    def read_logs(self, offset: int) -> dict[str, Any]:
        if not self._log.exists():
            return {"offset": 0, "content": ""}
        size = self._log.stat().st_size
        if offset > size:
            offset = 0
        with self._log.open("rb") as fh:
            fh.seek(offset)
            data = fh.read()
        return {"offset": size, "content": data.decode("utf-8", errors="replace")}


runner = JobRunner()


def start_install_aitoolkit() -> None:
    """Clone + install AI Toolkit (ostris) and its virtualenv."""
    script = APP_ROOT / "scripts" / "install_aitoolkit.sh"
    runner.start("Installing AI Toolkit + environment", ["bash", str(script)])


def start_install_musubi() -> None:
    """Clone + install Musubi-Tuner and its training virtualenv."""
    script = APP_ROOT / "scripts" / "install_musubi.sh"
    runner.start("Installing Musubi-Tuner + training environment", ["bash", str(script)])


def start_predownload_turbo() -> None:
    """Optionally pre-cache Krea-2-Turbo + the de-distill adapter from HF.

    Not required — AI Toolkit downloads these on the first run — but doing it up
    front avoids a long wait when training starts.
    """
    settings = config.load_settings()
    py = (
        "import os; from huggingface_hub import snapshot_download;"
        "tok=os.environ.get('TOKEN') or None;"
        "print('Downloading base model:', os.environ['BASE']);"
        "snapshot_download(os.environ['BASE'], token=tok);"
        "print('Downloading adapter:', os.environ['ADAPTER']);"
        "snapshot_download(os.environ['ADAPTER'], token=tok);"
        "print('Done. Cached to the Hugging Face hub cache.')"
    )
    cmd = ["bash", "-c",
           f'{sys.executable} -c "import huggingface_hub" 2>/dev/null || '
           f'{sys.executable} -m pip install -q huggingface_hub; '
           f'{sys.executable} -c "$PYCODE"']
    env = {"PYCODE": py, "BASE": settings["base_model"],
           "ADAPTER": settings["assistant_lora"], "TOKEN": settings.get("hf_token", ""),
           **config.hf_env(settings)}
    runner.start("Pre-downloading Krea-2-Turbo + adapter", cmd, env=env)


def start_download_model(target: str, url: str = "", hf_repo: str = "",
                         hf_file: str = "", hf_token: str = "") -> str:
    """Download a model file to ./models with the expected filename.

    `target` is "dit" or "vae"; either a direct `url` or a Hugging Face
    repo+file pair must be provided.
    """
    settings = config.load_settings()
    dest = Path(settings["dit_model"] if target == "dit" else settings["vae_model"])
    dest.parent.mkdir(parents=True, exist_ok=True)

    if url:
        cmd = ["bash", "-c", 'curl -L --fail --progress-bar -o "$DEST" "$URL"']
        env = {"DEST": str(dest), "URL": url}
        runner.start(f"Downloading {target.upper()} model", cmd, env=env)
        return str(dest)

    if hf_repo and hf_file:
        py = (
            "import os; from huggingface_hub import hf_hub_download;"
            "p=hf_hub_download(os.environ['REPO'], os.environ['FILE'],"
            "token=os.environ.get('TOKEN') or None);"
            "import shutil; shutil.copy(p, os.environ['DEST']);"
            "print('Saved to', os.environ['DEST'])"
        )
        # Ensure huggingface_hub is available, then run the download.
        cmd = ["bash", "-c",
               f'{sys.executable} -c "import huggingface_hub" 2>/dev/null || '
               f'{sys.executable} -m pip install -q huggingface_hub; '
               f'{sys.executable} -c "$PYCODE"']
        env = {"PYCODE": py, "REPO": hf_repo, "FILE": hf_file,
               "DEST": str(dest), "TOKEN": hf_token}
        runner.start(f"Downloading {target.upper()} model from Hugging Face", cmd, env=env)
        return str(dest)

    raise ValueError("Provide either a download URL or a Hugging Face repo + file.")
