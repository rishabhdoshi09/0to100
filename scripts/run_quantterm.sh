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
AUTONOMY_EXTERNAL=0

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

echo "[STACK] Starting local API at http://127.0.0.1:8765 …"
python -u -m uvicorn terminal_product_api:app --host 127.0.0.1 --port 8765 &
API_PID=$!

echo "[STACK] Waiting for terminal API (bhav load + market ops bootstrap can take ~15–45s)…"
API_READY=0
for _ in $(seq 1 240); do
  if ! kill -0 "$API_PID" >/dev/null 2>&1; then
    echo "[STACK] Terminal API exited during startup. Review errors above." >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "http://127.0.0.1:8765/api/health" >/dev/null 2>&1; then
    API_READY=1
    break
  fi
  sleep 0.5
done
if [[ "$API_READY" != "1" ]]; then
  echo "[STACK] Terminal API did not become ready within 120s." >&2
  exit 1
fi
echo "[STACK] Terminal API ready."

echo "[STACK] Starting dedicated terminal at http://127.0.0.1:5173 …"
(
  cd frontend
  npm run dev -- --host 127.0.0.1
) &
FRONTEND_PID=$!

echo "[STACK] QuantTerm is ready. Open http://127.0.0.1:5173 — keep this terminal open; Ctrl-C stops local child services."

while true; do
  for entry in "API:$API_PID" "FRONTEND:$FRONTEND_PID"; do
    name="${entry%%:*}"
    pid="${entry##*:}"
    if ! kill -0 "$pid" >/dev/null 2>&1; then
      echo "[STACK] $name process exited unexpectedly (pid=$pid)."
      exit 1
    fi
  done
  if [[ -n "$AUTONOMY_PID" ]] && ! kill -0 "$AUTONOMY_PID" >/dev/null 2>&1; then
    echo "[STACK] AUTONOMY process exited unexpectedly (pid=$AUTONOMY_PID)."
    exit 1
  fi
  sleep 3
done
