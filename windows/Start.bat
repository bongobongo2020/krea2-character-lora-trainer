@echo off
REM ============================================================
REM  Launch the LoRA Trainer and open the browser.
REM ============================================================
title LoRA Trainer
cd /d "%~dp0.."

if not exist ".webvenv\Scripts\python.exe" (
  echo Not installed yet. Please run windows\Install.bat first.
  pause
  exit /b 1
)

REM Open the UI in the default browser shortly after the server starts.
start "" /b cmd /c "timeout /t 3 >nul & start """" http://127.0.0.1:8000"

echo Starting LoRA Trainer at http://127.0.0.1:8000
echo (Other devices on your network can use http://<this-pc-ip>:8000)
echo Close this window to stop the app.
".webvenv\Scripts\python.exe" -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
pause
