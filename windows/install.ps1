# ============================================================
#  Krea 2 Character LoRA Trainer - Windows 11 installer
#  Run via Install.bat (double-click). No command line needed.
# ============================================================
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Step($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok($m)    { Write-Host "    [OK] $m"   -ForegroundColor Green }
function Warn($m)  { Write-Host "    [!]  $m"   -ForegroundColor Yellow }
function Info($m)  { Write-Host "    $m"        -ForegroundColor Gray }

Write-Host "============================================" -ForegroundColor Magenta
Write-Host "  Krea 2 Character LoRA Trainer - Setup" -ForegroundColor Magenta
Write-Host "============================================" -ForegroundColor Magenta

# --- 1. Dependencies: winget, git, python, uv ----------------------------
Step "Checking dependencies"

function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

$haveWinget = Have winget
if (-not $haveWinget) { Warn "winget not found - will try to use existing tools, otherwise install Git/Python manually." }

# Python 3.10-3.12 (prefer the 'py' launcher). Track exe + args separately so
# the bare "python" case (no version arg) invokes cleanly.
$PyExe = $null; $PyArgs = @()
foreach ($cand in @(@("py","-3.12"), @("py","-3.11"), @("py","-3.10"), @("python"))) {
    $exe = $cand[0]; $rest = @($cand | Select-Object -Skip 1)
    if (Have $exe) {
        try { & $exe @rest --version *> $null; if ($LASTEXITCODE -eq 0) { $PyExe = $exe; $PyArgs = $rest; break } } catch {}
    }
}
if (-not $PyExe) {
    if ($haveWinget) {
        Info "Installing Python 3.12 via winget..."
        winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
        Warn "Python was just installed. If the next step fails, close this window and run Install.bat again (so PATH refreshes)."
        $PyExe = "py"; $PyArgs = @("-3.12")
    } else { throw "Python 3.10-3.12 is required. Install from https://www.python.org/downloads/ and re-run." }
}
Ok ("Python: " + $PyExe + " " + ($PyArgs -join " "))

if (-not (Have git)) {
    if ($haveWinget) { Info "Installing Git via winget..."; winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements }
    else { throw "Git is required. Install from https://git-scm.com/download/win and re-run." }
}
Ok "Git present"

# uv (fast installer)
if (-not (Have uv)) {
    Info "Installing uv (fast Python package manager)..."
    try { if ($haveWinget) { winget install -e --id astral-sh.uv --accept-source-agreements --accept-package-agreements } } catch {}
    if (-not (Have uv)) {
        & $PyExe @PyArgs -m pip install --user uv
    }
}
$UV = (Have uv)
if ($UV) { Ok "uv present" } else { Warn "uv not available - falling back to pip (slower)." }

# --- 2. Choose / locate the models folder --------------------------------
Step "Where should models be stored?"
Add-Type -AssemblyName System.Windows.Forms | Out-Null
$dlg = New-Object System.Windows.Forms.FolderBrowserDialog
$dlg.Description = "Pick a folder to store (or reuse existing) AI models: Krea 2 Turbo, the adapter, and Qwen-VL. Existing downloads here will be reused."
$default = Join-Path $env:USERPROFILE "ai-models"
if (-not (Test-Path $default)) { New-Item -ItemType Directory -Force -Path $default | Out-Null }
$dlg.SelectedPath = $default
if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { $ModelsDir = $dlg.SelectedPath } else { $ModelsDir = $default }
$HfHome = Join-Path $ModelsDir "huggingface"
New-Item -ItemType Directory -Force -Path $HfHome | Out-Null
Ok "Models folder: $ModelsDir"

# --- 3. Scan for existing model downloads (avoid redownloads) -------------
Step "Scanning for already-downloaded models"
$hubDirs = @((Join-Path $HfHome "hub"), (Join-Path $env:USERPROFILE ".cache\huggingface\hub"))
function Find-Model($repo) {
    $name = "models--" + ($repo -replace "/","--")
    foreach ($h in $hubDirs) { $p = Join-Path $h $name; if (Test-Path $p) { return $p } }
    return $null
}
foreach ($m in @("krea/Krea-2-Turbo","ostris/krea2_turbo_training_adapter","Qwen/Qwen3-VL-4B-Instruct")) {
    if (Find-Model $m) { Ok "cached: $m" } else { Warn "not cached (will download on first use): $m" }
}

# --- 4. Detect VRAM + RAM and tune ---------------------------------------
Step "Detecting hardware"
$Vram = $null; $Gpu = $null
if (Have nvidia-smi) {
    try {
        $line = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | Select-Object -First 1)
        $p = $line.Split(","); $Gpu = $p[0].Trim(); $Vram = [int]($p[1].Trim())
    } catch {}
}
$RamGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if ($Gpu) { Ok "GPU: $Gpu  ($([math]::Round($Vram/1024,1)) GB VRAM)" } else { Warn "No NVIDIA GPU detected - training needs a CUDA GPU." }
Ok "System RAM: $RamGb GB"
if ($Vram -and $Vram -lt 12000) { Warn "Under 12 GB VRAM: training the 12B model will be slow/tight; the app auto-lowers resolution." }

# --- 5. Install the web server venv --------------------------------------
Step "Installing the app (web server)"
if (-not (Test-Path ".webvenv")) {
    & $PyExe @PyArgs -m venv .webvenv
}
& ".webvenv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".webvenv\Scripts\python.exe" -m pip install -r backend\requirements.txt
Ok "Web server ready"

# --- 6. Install AI Toolkit (Turbo engine) --------------------------------
Step "Installing AI Toolkit (Krea 2 Turbo engine) - this downloads several GB"
if (-not (Test-Path "ai-toolkit\.git")) {
    git clone https://github.com/ostris/ai-toolkit.git ai-toolkit
    git -C ai-toolkit submodule update --init --recursive
} else {
    git -C ai-toolkit pull --ff-only
    git -C ai-toolkit submodule update --init --recursive
}
# Install a list of packages into a venv's python (uv if available, else pip).
function PipInstall($pyPath, $argsLine) {
    $a = $argsLine.Split(" ")
    if ($UV) { uv pip install --python $pyPath @a } else { & $pyPath -m pip install @a }
}

$aiPy = "ai-toolkit\venv\Scripts\python.exe"
if (-not (Test-Path "ai-toolkit\venv")) {
    if ($UV) { uv venv --python 3.12 ai-toolkit\venv } else { & $PyExe @PyArgs -m venv ai-toolkit\venv }
}
Info "Installing PyTorch (CUDA 12.4) + torchaudio..."
PipInstall $aiPy "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124"
Info "Installing AI Toolkit requirements..."
PipInstall $aiPy "-r ai-toolkit\requirements.txt"
Ok "AI Toolkit ready"

# --- 6b. Optional: musubi-tuner (Raw engine) -----------------------------
Step "Optional second engine: musubi-tuner (Krea 2 Raw)"
Info "The Raw engine trains on the Krea 2 raw model (you supply the raw + VAE"
Info "files). The Turbo engine you just installed is recommended and enough for"
Info "most users. You can always add this later by re-running the installer."
$ans = [System.Windows.Forms.MessageBox]::Show(
    "Also install the optional musubi-tuner (Krea 2 Raw) engine?`n`nThis downloads another PyTorch environment (several GB). Choose No to skip - the Turbo engine is already installed.",
    "Install Raw engine?", [System.Windows.Forms.MessageBoxButtons]::YesNo,
    [System.Windows.Forms.MessageBoxIcon]::Question)

if ($ans -eq [System.Windows.Forms.DialogResult]::Yes) {
    if (-not (Test-Path "musubi-tuner\.git")) {
        git clone https://github.com/kohya-ss/musubi-tuner.git musubi-tuner
    } else {
        git -C musubi-tuner pull --ff-only
    }
    $muPy = "musubi-tuner\.venv\Scripts\python.exe"
    if (-not (Test-Path "musubi-tuner\.venv")) {
        if ($UV) { uv venv --python 3.11 musubi-tuner\.venv } else { & $PyExe @PyArgs -m venv musubi-tuner\.venv }
    }
    Info "Installing PyTorch (CUDA 12.4)..."
    PipInstall $muPy "torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124"
    Info "Installing musubi-tuner package + accelerate..."
    # musubi-tuner is a pyproject package (no requirements.txt) -> editable install.
    Push-Location musubi-tuner
    if ($UV) { uv pip install --python ".venv\Scripts\python.exe" -e . } else { & ".venv\Scripts\python.exe" -m pip install -e . }
    Pop-Location
    PipInstall $muPy "accelerate"
    $InstalledMusubi = $true
    Ok "musubi-tuner ready (supply krea2_raw.safetensors + the VAE in the app's Setup tab)"
} else {
    $InstalledMusubi = $false
    Info "Skipped musubi-tuner. (Turbo engine is installed and ready.)"
}

# --- 7. Write settings (paths + models folder + tuning) ------------------
Step "Saving settings"
New-Item -ItemType Directory -Force -Path "workspace\projects" | Out-Null
$settings = [ordered]@{
    aitoolkit_dir    = (Join-Path $Root "ai-toolkit")
    aitoolkit_python = (Join-Path $Root "ai-toolkit\venv\Scripts\python.exe")
    base_model       = "krea/Krea-2-Turbo"
    assistant_lora   = "ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors"
    hf_home          = $HfHome
    caption_model    = "Qwen/Qwen3-VL-4B-Instruct"
    auto_free_vram   = "true"
}
if ($InstalledMusubi) {
    $settings.musubi_dir     = (Join-Path $Root "musubi-tuner")
    $settings.accelerate_bin = (Join-Path $Root "musubi-tuner\.venv\Scripts\accelerate.exe")
}
$settings | ConvertTo-Json | Set-Content -Encoding UTF8 "workspace\settings.json"
Ok "Settings written (models folder: $ModelsDir)"

# --- 8. Desktop shortcut --------------------------------------------------
Step "Creating desktop shortcut"
try {
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path ([Environment]::GetFolderPath("Desktop")) "Krea 2 LoRA Trainer.lnk"))
    $lnk.TargetPath = (Join-Path $PSScriptRoot "Start.bat")
    $lnk.WorkingDirectory = $Root
    $lnk.IconLocation = "shell32.dll,13"
    $lnk.Save()
    Ok "Desktop shortcut created"
} catch { Warn "Could not create desktop shortcut: $_" }

Write-Host "`n============================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  Launch with the 'Krea 2 LoRA Trainer' desktop icon," -ForegroundColor Green
Write-Host "  or run windows\Start.bat. Your browser opens automatically." -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
