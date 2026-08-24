#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d venv ]]; then
  echo "Missing venv. Create the QuantTerm Python environment first." >&2
  exit 1
fi

source venv/bin/activate

if ! python -c 'import fastapi, uvicorn' >/dev/null 2>&1; then
  echo "[STACK] Installing local terminal API dependencies…"
  python -m pip install 'fastapi>=0.115.0' 'uvicorn>=0.30.0'
fi

if [[ ! -d frontend/node_modules ]]; then
  echo "[STACK] Installing terminal frontend dependencies…"
  (cd frontend && npm install)
fi

port_open() {
  local port="$1"
  python - "$port" <<'PY'
import socket, sys
port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.4)
try:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
finally:
    sock.close()
PY
}

url_ok() {
  local url="$1"
  python - "$url" <<'PY'
import sys, urllib.error, urllib.request
url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=1.5) as response:
        raise SystemExit(0 if int(response.status) == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

AUTONOMY_PID=""
API_PID=""
FRONTEND_PID=""
AUTONOMY_EXTERNAL=0
API_EXTERNAL=0
FRONTEND_EXTERNAL=0

cleanup() {
  echo
  echo "[STACK] Stopping QuantTerm child services…"
  for pid in "$FRONTEND_PID" "$API_PID" "$AUTONOMY_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait >/dev/null 2>&1 || true
  if [[ "$AUTONOMY_EXTERNAL" == "1" ]]; then
    echo "[STACK] Existing external autonomy supervisor was left running."
  fi
  if [[ "$API_EXTERNAL" == "1" ]]; then
    echo "[STACK] Existing market API on :8765 was left running."
  fi
  if [[ "$FRONTEND_EXTERNAL" == "1" ]]; then
    echo "[STACK] Existing RecoWealth desk on :5173 was left running."
  fi
  echo "[STACK] Child services stopped."
}
trap cleanup EXIT INT TERM

if python - <<'PY' >/dev/null 2>&1
from product.autonomy_status import read_autonomy_status
raise SystemExit(0 if read_autonomy_status().get("running") else 1)
PY
then
  AUTONOMY_EXTERNAL=1
  echo "[STACK] A healthy autonomy supervisor is already running; reusing it."
else
  echo "[STACK] Starting autonomy supervisor…"
  python -u main.py autonomy &
  AUTONOMY_PID=$!
  sleep 1
  if ! kill -0 "$AUTONOMY_PID" >/dev/null 2>&1; then
    if python - <<'PY' >/dev/null 2>&1
from product.autonomy_status import read_autonomy_status
raise SystemExit(0 if read_autonomy_status().get("running") else 1)
PY
    then
      echo "[STACK] Another healthy supervisor acquired the lock; reusing it."
      AUTONOMY_PID=""
      AUTONOMY_EXTERNAL=1
    else
      echo "[STACK] Autonomy failed to stay alive. Review the visible error above."
      exit 1
    fi
  fi
fi

if url_ok "http://127.0.0.1:8765/api/health"; then
  API_EXTERNAL=1
  echo "[STACK] Reusing market API at http://127.0.0.1:8765"
elif port_open 8765; then
  echo "[STACK] Port 8765 is occupied but /api/health is not ready." >&2
  exit 1
else
  echo "[STACK] Starting local API at http://127.0.0.1:8765 …"
  python -u -m uvicorn terminal_product_api:app --host 127.0.0.1 --port 8765 &
  API_PID=$!
fi

if port_open 5173; then
  FRONTEND_EXTERNAL=1
  echo "[STACK] Reusing dedicated terminal at http://127.0.0.1:5173"
else
  echo "[STACK] Starting dedicated terminal at http://127.0.0.1:5173 …"
  (
    cd frontend
    npm run dev -- --host 127.0.0.1
  ) &
  FRONTEND_PID=$!
fi

echo "[STACK] QuantTerm is starting. Keep this terminal open; Ctrl-C stops local child services."

while true; do
  if [[ -n "$API_PID" ]] && ! kill -0 "$API_PID" >/dev/null 2>&1; then
    echo "[STACK] API process exited unexpectedly (pid=$API_PID)."
    exit 1
  fi
  if [[ -n "$FRONTEND_PID" ]] && ! kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    echo "[STACK] FRONTEND process exited unexpectedly (pid=$FRONTEND_PID)."
    exit 1
  fi
  if [[ -n "$AUTONOMY_PID" ]] && ! kill -0 "$AUTONOMY_PID" >/dev/null 2>&1; then
    echo "[STACK] AUTONOMY process exited unexpectedly (pid=$AUTONOMY_PID)."
    exit 1
  fi
  sleep 3
done
