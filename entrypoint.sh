#!/bin/sh
set -eu

cd /opt/flux3/app
python download_models.py

rm -rf /opt/ComfyUI/user/default/workflows
mkdir -p /opt/ComfyUI/user/default/workflows
cp -f /opt/flux3/workflows/*.json /opt/ComfyUI/user/default/workflows/

cd /opt/ComfyUI
python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch --disable-dynamic-vram &
comfy_pid=$!
trap 'kill "$comfy_pid"' EXIT INT TERM

ready=0
i=0
while [ "$i" -lt 180 ]; do
  if python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)' >/dev/null 2>&1; then
    ready=1
    break
  fi
  i=$((i + 1))
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "ComfyUI did not open port 8188" >&2
  exit 1
fi

cd /opt/flux3/app
: "${GENERATION_TIMEOUT:=1800}"
export GENERATION_TIMEOUT
python server.py
