@echo off
REM ============================================================
REM  LoRA Trainer - Windows 11 installer
REM  Double-click this file. No command line needed.
REM ============================================================
title LoRA Trainer - Installer
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
echo.
echo Installer finished. You can close this window.
pause
