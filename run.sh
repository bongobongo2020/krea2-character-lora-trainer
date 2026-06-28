#!/bin/bash
# Start the Krea2 LoRA Trainer web app (backend + frontend on one port).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Create a lightweight venv just for the web server (separate from the heavy
# training venv inside musubi-tuner/.venv).
if [ ! -d ".webvenv" ]; then
  echo "==> Creating web-server virtualenv…"
  python3 -m venv .webvenv
  ./.webvenv/bin/pip install --upgrade pip >/dev/null
  ./.webvenv/bin/pip install -r backend/requirements.txt
fi

# Bind to all interfaces by default so other machines on your LAN can reach it.
# Set HOST=127.0.0.1 to restrict access to this machine only.
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# Best-effort: show the LAN address other devices should open.
LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo "==> Krea2 LoRA Trainer starting on $HOST:$PORT"
echo "    Local:   http://127.0.0.1:$PORT"
[ -n "$LAN_IP" ] && echo "    Network: http://$LAN_IP:$PORT   (open this from other LAN devices)"
echo "    If a firewall is on, allow the port, e.g.:  sudo ufw allow $PORT/tcp"
exec ./.webvenv/bin/uvicorn backend.main:app --host "$HOST" --port "$PORT"
