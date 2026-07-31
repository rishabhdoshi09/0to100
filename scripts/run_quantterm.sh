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

AUTONOMY_PID=""
API_PID=""
FRONTEND_PID=""

cleanup() {
  echo
  echo "[STACK] Stopping QuantTerm services…"
  for pid in "$FRONTEND_PID" "$API_PID" "$AUTONOMY_PID"; do
    if [[ -n "$pid" ]]; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  wait >/dev/null 2>&1 || true
  echo "[STACK] All child services stopped."
}
trap cleanup EXIT INT TERM

echo "[STACK] Starting autonomy supervisor…"
python -u main.py autonomy &
AUTONOMY_PID=$!

echo "[STACK] Starting local API at http://127.0.0.1:8765 …"
python -u -m uvicorn terminal_api:app --host 127.0.0.1 --port 8765 &
API_PID=$!

echo "[STACK] Starting dedicated terminal at http://127.0.0.1:5173 …"
(
  cd frontend
  npm run dev -- --host 127.0.0.1
) &
FRONTEND_PID=$!

echo "[STACK] QuantTerm is starting. Keep this terminal open; Ctrl-C stops the full local stack."

while true; do
  for entry in "AUTONOMY:$AUTONOMY_PID" "API:$API_PID" "FRONTEND:$FRONTEND_PID"; do
    name="${entry%%:*}"
    pid="${entry##*:}"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "[STACK] $name process exited unexpectedly (pid=$pid)."
      exit 1
    fi
  done
  sleep 3
done
