"""FastAPI server: REST API + static frontend for the Krea2 LoRA trainer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, trainer, setup_tasks, captioning
from .trainer import Project, manager

APP_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = APP_ROOT / "frontend"

app = FastAPI(title="Krea2 Character LoRA Trainer")


# --- settings ------------------------------------------------------------
class SettingsIn(BaseModel):
    # AI Toolkit (Turbo)
    aitoolkit_dir: str | None = None
    aitoolkit_python: str | None = None
    base_model: str | None = None
    assistant_lora: str | None = None
    hf_token: str | None = None
    hf_home: str | None = None
    caption_model: str | None = None
    caption_python: str | None = None
    auto_free_vram: str | None = None
    # musubi-tuner (Raw)
    musubi_dir: str | None = None
    accelerate_bin: str | None = None
    dit_model: str | None = None
    vae_model: str | None = None


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    settings = config.load_settings()
    return {"settings": settings, "environment": config.check_environment(settings)}


@app.post("/api/settings")
def update_settings(payload: SettingsIn) -> dict[str, Any]:
    settings = config.save_settings(payload.model_dump(exclude_none=True))
    return {"settings": settings, "environment": config.check_environment(settings)}


# --- projects ------------------------------------------------------------
class ProjectIn(BaseModel):
    name: str
    engine: str = trainer.DEFAULT_ENGINE


@app.get("/api/projects")
def get_projects() -> dict[str, Any]:
    return {"projects": trainer.list_projects()}


@app.post("/api/projects")
def post_project(payload: ProjectIn) -> dict[str, Any]:
    if not payload.name.strip():
        raise HTTPException(400, "Project name is required.")
    proj = trainer.create_project(payload.name, engine=payload.engine)
    return _project_view(proj)


def _require_project(project_id: str) -> Project:
    proj = Project.load(project_id)
    if not proj:
        raise HTTPException(404, "Project not found.")
    return proj


def _project_view(proj: Project) -> dict[str, Any]:
    return {
        "id": proj.id,
        "name": proj.name,
        "engine": proj.engine,
        "params": proj.params,
        "images": proj.list_images(),
        "outputs": proj.list_outputs(),
        "output_name": proj.output_name(),
    }


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    return _project_view(_require_project(project_id))


class ParamsIn(BaseModel):
    params: dict[str, Any]


@app.put("/api/projects/{project_id}/params")
def put_params(project_id: str, payload: ParamsIn) -> dict[str, Any]:
    proj = _require_project(project_id)
    allowed = trainer.default_params(proj.engine)
    for k, v in payload.params.items():
        if k in allowed:
            proj.params[k] = v
    proj.save()
    return _project_view(proj)


# --- dataset: images + captions -----------------------------------------
@app.post("/api/projects/{project_id}/images")
async def upload_images(project_id: str, files: list[UploadFile] = File(...)) -> dict[str, Any]:
    proj = _require_project(project_id)
    proj.ensure_dirs()
    saved = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in trainer.IMAGE_EXTS:
            continue
        # Sanitize the filename, keep it stable for caption pairing.
        stem = Path(f.filename).stem
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem) or "img"
        dest = proj.images_dir / f"{safe}{ext}"
        i = 1
        while dest.exists():
            dest = proj.images_dir / f"{safe}_{i}{ext}"
            i += 1
        dest.write_bytes(await f.read())
        # Seed an empty caption file with the trigger word for convenience.
        caption_file = dest.with_suffix(".txt")
        if not caption_file.exists():
            caption_file.write_text(str(proj.params.get("trigger_word", "")))
        saved.append(dest.name)
    return {"saved": saved, "images": proj.list_images()}


@app.post("/api/projects/{project_id}/import_zip")
async def import_zip(project_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    proj = _require_project(project_id)
    proj.ensure_dirs()
    try:
        saved = proj.import_zip(await file.read(),
                                default_caption=str(proj.params.get("trigger_word", "")))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"saved": saved, "images": proj.list_images()}


class CaptionsIn(BaseModel):
    captions: dict[str, str]  # filename -> caption text


@app.put("/api/projects/{project_id}/captions")
def put_captions(project_id: str, payload: CaptionsIn) -> dict[str, Any]:
    proj = _require_project(project_id)
    for filename, text in payload.captions.items():
        img = proj.images_dir / filename
        if img.exists() and img.suffix.lower() in trainer.IMAGE_EXTS:
            img.with_suffix(".txt").write_text(text.strip())
    return {"images": proj.list_images()}


class AutocaptionIn(BaseModel):
    overwrite: bool = False


@app.post("/api/projects/{project_id}/autocaption")
def start_autocaption(project_id: str, payload: AutocaptionIn) -> dict[str, Any]:
    proj = _require_project(project_id)
    settings = config.load_settings()
    # Validate first, so a doomed run never disturbs ComfyUI.
    if captioning.runner.is_running():
        raise HTTPException(409, "A captioning run is already in progress.")
    if not proj.list_images():
        raise HTTPException(409, "No images to caption — upload some first.")
    auto = auto_free_vram(settings, VRAM_NEED_MB["caption"])
    try:
        result = captioning.start(proj, settings, payload.overwrite)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(409, str(e))
    result["auto_free"] = auto
    return result


@app.get("/api/projects/{project_id}/autocaption/status")
def autocaption_status(project_id: str) -> dict[str, Any]:
    _require_project(project_id)
    return captioning.runner.status()


@app.get("/api/projects/{project_id}/autocaption/logs")
def autocaption_logs(project_id: str, offset: int = 0) -> dict[str, Any]:
    _require_project(project_id)
    return captioning.runner.read_logs(offset)


@app.delete("/api/projects/{project_id}/images/{filename}")
def delete_image(project_id: str, filename: str) -> dict[str, Any]:
    proj = _require_project(project_id)
    img = proj.images_dir / Path(filename).name
    if img.exists():
        img.unlink()
        cap = img.with_suffix(".txt")
        if cap.exists():
            cap.unlink()
    return {"images": proj.list_images()}


@app.get("/api/projects/{project_id}/images/{filename}")
def get_image(project_id: str, filename: str) -> FileResponse:
    proj = _require_project(project_id)
    img = proj.images_dir / Path(filename).name
    if not img.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(img)


# --- training ------------------------------------------------------------
@app.post("/api/projects/{project_id}/train")
def start_training(project_id: str) -> dict[str, Any]:
    proj = _require_project(project_id)
    settings = config.load_settings()
    # Validate first, so a doomed start never disturbs ComfyUI.
    if manager.is_running():
        raise HTTPException(409, "A training run is already in progress.")
    problems = manager.validate(proj, settings)
    if problems:
        raise HTTPException(400, "; ".join(problems))
    # Only now free VRAM (gentle ComfyUI unload) if the card is too full.
    auto = auto_free_vram(settings, VRAM_NEED_MB.get(proj.engine, 12000))
    try:
        result = manager.start(proj, settings)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    result["auto_free"] = auto
    return result


@app.post("/api/projects/{project_id}/stop")
def stop_training(project_id: str) -> dict[str, Any]:
    manager.stop()
    return {"stopped": True}


@app.get("/api/projects/{project_id}/status")
def training_status(project_id: str) -> dict[str, Any]:
    proj = _require_project(project_id)
    return manager.status(proj)


@app.get("/api/projects/{project_id}/logs")
def training_logs(project_id: str, offset: int = 0) -> dict[str, Any]:
    proj = _require_project(project_id)
    log = proj.log_file
    if not log.exists():
        return {"offset": 0, "content": "", "size": 0}
    size = log.stat().st_size
    if offset > size:  # log was truncated/restarted
        offset = 0
    with log.open("rb") as fh:
        fh.seek(offset)
        data = fh.read()
    return {"offset": size, "content": data.decode("utf-8", errors="replace"), "size": size}


@app.get("/api/projects/{project_id}/preview_command")
def preview_command(project_id: str) -> dict[str, Any]:
    import shlex
    proj = _require_project(project_id)
    settings = config.load_settings()
    argv, cwd, _env = proj.build_command(settings)
    cmd = " \\\n  ".join(shlex.quote(c) for c in argv)
    if proj.engine == "aitoolkit":
        config_text = proj.yaml_file.read_text()
        config_label = "config.yaml"
    else:
        config_text = proj.toml_file.read_text()
        config_label = "dataset.toml"
    return {"command": f"cd {shlex.quote(cwd)} && \\\n  {cmd}",
            "config_label": config_label, "config": config_text}


@app.get("/api/projects/{project_id}/samples")
def list_samples(project_id: str) -> dict[str, Any]:
    proj = _require_project(project_id)
    return {"groups": proj.list_samples()}


@app.get("/api/projects/{project_id}/sample/{relpath:path}")
def get_sample(project_id: str, relpath: str) -> FileResponse:
    proj = _require_project(project_id)
    f = proj.find_output(relpath)  # samples live under output_dir; same guard
    if not f:
        raise HTTPException(404, "Sample not found.")
    return FileResponse(f)  # served inline for display in the browser


@app.get("/api/projects/{project_id}/outputs/{relpath:path}")
def download_output(project_id: str, relpath: str) -> FileResponse:
    proj = _require_project(project_id)
    f = proj.find_output(relpath)
    if not f:
        raise HTTPException(404, "File not found.")
    return FileResponse(f, filename=f.name, media_type="application/octet-stream")


# --- setup jobs (UI-driven install / model download) --------------------
class DownloadIn(BaseModel):
    target: str  # "dit" or "vae"
    url: str = ""
    hf_repo: str = ""
    hf_file: str = ""
    hf_token: str = ""


@app.post("/api/setup/install_aitoolkit")
def setup_install_aitoolkit() -> dict[str, Any]:
    try:
        setup_tasks.start_install_aitoolkit()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"started": True}


@app.post("/api/setup/predownload_turbo")
def setup_predownload_turbo() -> dict[str, Any]:
    try:
        setup_tasks.start_predownload_turbo()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"started": True}


@app.post("/api/setup/install_musubi")
def setup_install_musubi() -> dict[str, Any]:
    try:
        setup_tasks.start_install_musubi()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"started": True}


@app.post("/api/setup/download_model")
def setup_download_model(payload: DownloadIn) -> dict[str, Any]:
    if payload.target not in ("dit", "vae"):
        raise HTTPException(400, "target must be 'dit' or 'vae'.")
    try:
        dest = setup_tasks.start_download_model(
            payload.target, url=payload.url.strip(), hf_repo=payload.hf_repo.strip(),
            hf_file=payload.hf_file.strip(), hf_token=payload.hf_token.strip())
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409, str(e))
    return {"started": True, "dest": dest}


@app.get("/api/setup/model_suggestions")
def model_suggestions() -> dict[str, Any]:
    """Known filenames + candidate Hugging Face repos for the auto-fill button.

    Filenames are canonical (the trainer expects exactly these names). Repos are
    best-effort suggestions the user can pick from or override — the Krea 2 base
    model is gated and may live in a repo you have private access to.
    """
    return {
        "dit": {
            "filename": "raw.safetensors",
            "repos": ["krea/Krea-2-Raw"],
            "note": "The Krea 2 Raw MMDiT weights (the file is named raw.safetensors in the repo).",
        },
        "vae": {
            "filename": "vae/diffusion_pytorch_model.safetensors",
            "repos": ["krea/Krea-2-Turbo", "Comfy-Org/Qwen-Image_ComfyUI", "Qwen/Qwen-Image"],
            "note": "Qwen-Image VAE (also bundled in the Krea-2-Turbo repo).",
        },
    }


@app.get("/api/setup/status")
def setup_status() -> dict[str, Any]:
    return setup_tasks.runner.status()


@app.get("/api/setup/logs")
def setup_logs(offset: int = 0) -> dict[str, Any]:
    return setup_tasks.runner.read_logs(offset)


# Core OS/compositor processes that legitimately hold GPU memory but must
# never be offered an "End process" button (lowercased, Windows).
_PROTECTED_PROCS = {
    "system", "registry", "dwm.exe", "csrss.exe", "winlogon.exe", "wininit.exe",
    "services.exe", "lsass.exe", "smss.exe", "fontdrvhost.exe", "sihost.exe",
    "explorer.exe", "ctfmon.exe", "taskhostw.exe", "searchhost.exe",
    "shellexperiencehost.exe", "startmenuexperiencehost.exe", "textinputhost.exe",
}


def _windows_pid_names(pids: set[int]) -> dict[int, str]:
    """Resolve PID -> image name on Windows via tasklist."""
    import csv
    import io
    import subprocess
    try:
        out = subprocess.run(["tasklist", "/fo", "csv", "/nh"],
                             capture_output=True, text=True, timeout=10)
    except Exception:
        return {}
    names: dict[int, str] = {}
    for row in csv.reader(io.StringIO(out.stdout)):
        if len(row) >= 2:
            try:
                pid = int(row[1])
            except ValueError:
                continue
            if pid in pids:
                names[pid] = row[0]
    return names


_win_procs_cache: dict[str, Any] = {"at": 0.0, "procs": []}


def _windows_gpu_procs() -> list[dict[str, Any]]:
    """Per-process GPU memory on Windows, where nvidia-smi reports no compute
    apps under the consumer WDDM driver. Reads the same perf counter Task
    Manager uses (\\GPU Process Memory(*)\\Dedicated Usage), then resolves names.
    Cached for a few seconds — typeperf takes ~1-2s and /api/gpu is polled."""
    import re
    import subprocess
    import time
    if time.time() - _win_procs_cache["at"] < 5:
        return _win_procs_cache["procs"]
    try:
        out = subprocess.run(
            ["typeperf", r"\GPU Process Memory(*)\Dedicated Usage", "-sc", "1"],
            capture_output=True, text=True, timeout=15)
        used: dict[int, int] = {}
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        header = next((ln for ln in lines if ln.startswith('"(PDH-CSV')), None)
        if header is not None and lines.index(header) + 1 < len(lines):
            cols = [c.strip('"') for c in header.split('","')]
            vals = [c.strip('"') for c in lines[lines.index(header) + 1].split('","')]
            for col, val in zip(cols[1:], vals[1:]):  # skip the timestamp column
                m = re.search(r"pid_(\d+)_", col)
                if not m:
                    continue
                try:
                    b = float(val)
                except ValueError:
                    continue
                if b > 0:
                    used[int(m.group(1))] = used.get(int(m.group(1)), 0) + int(b)
        names = _windows_pid_names(set(used)) if used else {}
        procs = []
        for pid, b in sorted(used.items(), key=lambda kv: -kv[1]):
            name = names.get(pid, f"pid {pid}")
            procs.append({"pid": str(pid), "name": name, "used_mb": int(b // (1024 * 1024)),
                          "system": name.lower() in _PROTECTED_PROCS})
    except Exception:
        procs = []
    _win_procs_cache["at"] = time.time()
    _win_procs_cache["procs"] = procs
    return procs


def _gpu_status() -> dict[str, Any]:
    """Free/used VRAM + processes holding it (no port enrichment — hot path)."""
    import os
    import shutil
    import subprocess
    if not shutil.which("nvidia-smi"):
        return {"available": False}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        gpus = []
        for line in q.stdout.strip().splitlines():
            name, total, used, free = [x.strip() for x in line.split(",")]
            gpus.append({"name": name, "total_mb": int(total),
                         "used_mb": int(used), "free_mb": int(free)})
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        procs = []
        for line in p.stdout.strip().splitlines():
            if not line.strip():
                continue
            pid, pname, mem = [x.strip() for x in line.split(",")]
            procs.append({"pid": pid, "name": pname.split("/")[-1], "used_mb": int(mem)})
        # WDDM (consumer Windows) can't list compute apps; fall back to the
        # Windows perf counter so a VRAM-hogging server is still visible.
        if not procs and os.name == "nt" and gpus and gpus[0]["used_mb"] > 800:
            procs = _windows_gpu_procs()
        return {"available": True, "gpus": gpus, "processes": procs}
    except Exception:
        return {"available": False}


@app.get("/api/gpu")
def gpu_status() -> dict[str, Any]:
    """Free/used VRAM + processes holding it, so the UI can warn before OOM.
    Enriches each process with the TCP ports it listens on (helps identify a
    stray model server, e.g. one on :8080)."""
    s = _gpu_status()
    for pr in s.get("processes", []):
        try:
            pr["ports"] = _pid_listen_ports(int(pr["pid"]))
        except Exception:
            pr["ports"] = []
    return s


def _gpu_pids() -> dict[int, dict[str, Any]]:
    """Map of pid -> info for processes currently holding VRAM."""
    status = _gpu_status()
    return {int(p["pid"]): p for p in status.get("processes", [])}


def _pid_listen_ports(pid: int) -> list[int]:
    import os
    import subprocess
    ports: set[int] = set()
    try:
        if os.name == "nt":
            out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                                 capture_output=True, text=True, timeout=8)
            for line in out.stdout.splitlines():
                parts = line.split()
                if (len(parts) >= 5 and parts[0].upper() == "TCP"
                        and parts[-1] == str(pid) and parts[3].upper() == "LISTENING"):
                    try:
                        ports.add(int(parts[1].rsplit(":", 1)[-1]))
                    except ValueError:
                        pass
        else:
            out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=8)
            for line in out.stdout.splitlines():
                if f"pid={pid}," in line:
                    try:
                        ports.add(int(line.split()[3].rsplit(":", 1)[-1]))
                    except ValueError:
                        pass
    except Exception:
        pass
    return sorted(ports)


def _gpu_free_mb() -> int | None:
    g = _gpu_status()
    if g.get("available") and g.get("gpus"):
        return g["gpus"][0]["free_mb"]
    return None


def _comfy_unload(pid: int) -> int | None:
    """Gently free a process's VRAM via ComfyUI's /free API. Returns the port
    it worked on, or None if no reachable /free endpoint was found."""
    import json as _json
    import urllib.request
    data = _json.dumps({"unload_models": True, "free_memory": True}).encode()
    for port in _pid_listen_ports(pid):
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/free", data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=6) as r:
                if 200 <= r.status < 300:
                    return port
        except Exception:
            continue
    return None


# VRAM (MB) each engine/task needs free before it will reliably run.
VRAM_NEED_MB = {"aitoolkit": 18000, "musubi": 11000, "caption": 4000}


def auto_free_vram(settings: dict[str, Any], need_mb: int) -> dict[str, Any]:
    """If enabled and VRAM is short, gently unload ComfyUI-like processes until
    enough is free. Never kills processes — that stays a manual action."""
    enabled = str(settings.get("auto_free_vram", "true")).lower() in ("1", "true", "yes", "on")
    free = _gpu_free_mb()
    if not enabled or free is None or free >= need_mb:
        return {"attempted": False, "free_mb": free, "need_mb": need_mb, "actions": []}

    import time
    actions = []
    # Largest VRAM users first.
    for p in sorted(_gpu_status().get("processes", []), key=lambda x: -x["used_mb"]):
        if (_gpu_free_mb() or 0) >= need_mb:
            break
        port = _comfy_unload(int(p["pid"]))
        if port:
            time.sleep(1.5)
            actions.append({"pid": p["pid"], "name": p["name"],
                            "method": f"ComfyUI /free on :{port}"})
    free_after = _gpu_free_mb()
    return {"attempted": True, "free_mb": free_after, "need_mb": need_mb,
            "actions": actions, "enough": (free_after or 0) >= need_mb}


class FreeVramIn(BaseModel):
    pid: int
    mode: str = "comfy"  # "comfy" (gentle /free API) or "kill"


def _terminate_pid(pid: int) -> None:
    """End a process by PID, cross-platform. Raises PermissionError if the OS
    refuses (e.g. an elevated/other-owner process)."""
    import os
    import time
    if os.name == "nt":
        # SIGTERM/SIGKILL don't exist on Windows; taskkill is the reliable path
        # (/T also ends child processes, e.g. a server's worker subprocess).
        import subprocess
        r = subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            msg = (r.stderr or r.stdout or "").strip()
            low = msg.lower()
            if "denied" in low or "access" in low:
                raise PermissionError(msg)
            if "not found" in low or "no running" in low:
                return  # already gone — treat as success
            raise RuntimeError(msg or "taskkill failed")
        return
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


@app.post("/api/gpu/free")
def gpu_free(payload: FreeVramIn) -> dict[str, Any]:
    """Free VRAM held by a GPU process — gently via ComfyUI's /free API, or by
    ending the process. Only PIDs that currently hold VRAM may be targeted."""
    import time

    procs = _gpu_pids()
    if payload.pid not in procs:
        raise HTTPException(400, "That PID is not currently using the GPU.")
    if procs[payload.pid].get("system"):
        raise HTTPException(400, "That is a core system process and won't be ended from here.")
    before = _gpu_status()["gpus"][0]["free_mb"]

    if payload.mode == "comfy":
        port = _comfy_unload(payload.pid)
        if port is not None:
            time.sleep(1.5)
            after = _gpu_status()["gpus"][0]["free_mb"]
            return {"ok": True, "method": f"ComfyUI /free on :{port}",
                    "freed_mb": after - before, "free_mb": after}
        raise HTTPException(
            502, "Could not reach a ComfyUI /free endpoint for that process. "
                 "Use 'End process' instead, or unload from ComfyUI directly.")

    # mode == "kill"
    try:
        _terminate_pid(payload.pid)
    except PermissionError:
        raise HTTPException(403, "Not allowed to end that process — run the app as the "
                                 "same user (or as administrator).")
    except Exception as e:
        raise HTTPException(500, f"Could not end the process: {e}")
    time.sleep(1.0)
    after = _gpu_status()["gpus"][0]["free_mb"]
    return {"ok": True, "method": "ended process", "freed_mb": after - before, "free_mb": after}


@app.get("/api/hardware")
def hardware() -> dict[str, Any]:
    """Detected GPU/VRAM/RAM + the auto-tuned params per engine."""
    hw = config.detect_hardware()
    return {
        "hardware": hw,
        "recommendations": {
            "aitoolkit": trainer.recommend_params("aitoolkit", hw.get("vram_mb")),
            "musubi": trainer.recommend_params("musubi", hw.get("vram_mb")),
        },
    }


@app.get("/api/models/scan")
def models_scan() -> dict[str, Any]:
    """Report which models are already cached (avoids wasteful redownloads)."""
    return config.scan_models(config.load_settings())


@app.get("/api/global_status")
def global_status() -> dict[str, Any]:
    return {"running": manager.is_running(), "active_project": manager.active_project()}


# --- static frontend (mounted last so /api/* wins) -----------------------
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:  # pragma: no cover
    @app.get("/")
    def _no_frontend() -> JSONResponse:
        return JSONResponse({"error": "frontend directory missing"}, status_code=500)
