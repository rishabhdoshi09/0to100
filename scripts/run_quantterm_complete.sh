#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d venv ]]; then
  echo "Missing venv. Create the QuantTerm Python environment first." >&2
  exit 1
fi

source venv/bin/activate

if ! python -c 'import reportlab, fastapi, uvicorn' >/dev/null 2>&1; then
  echo "[STACK] Installing professional report dependencies…"
  python -m pip install 'reportlab>=4.2.0' 'fastapi>=0.115.0' 'uvicorn>=0.30.0'
fi

REPORT_PID=""
STACK_PID=""

cleanup() {
  echo
  echo "[COMPLETE STACK] Stopping report API and QuantTerm stack…"
  if [[ -n "$STACK_PID" ]]; then
    kill "$STACK_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$REPORT_PID" ]]; then
    kill "$REPORT_PID" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
  echo "[COMPLETE STACK] Stopped."
}
trap cleanup EXIT INT TERM

echo "[COMPLETE STACK] Starting research-report API at http://127.0.0.1:8766 …"
python -u -m uvicorn report_api:app --host 127.0.0.1 --port 8766 &
REPORT_PID=$!

sleep 1
if ! kill -0 "$REPORT_PID" >/dev/null 2>&1; then
  echo "[COMPLETE STACK] Research-report API failed to start. Review the error above." >&2
  exit 1
fi

echo "[COMPLETE STACK] Starting QuantTerm terminal, market operations and autonomy…"
bash scripts/run_quantterm.sh &
STACK_PID=$!

while true; do
  if ! kill -0 "$REPORT_PID" >/dev/null 2>&1; then
    echo "[COMPLETE STACK] REPORT API exited unexpectedly (pid=$REPORT_PID)." >&2
    exit 1
  fi
  if ! kill -0 "$STACK_PID" >/dev/null 2>&1; then
    echo "[COMPLETE STACK] QuantTerm stack exited unexpectedly (pid=$STACK_PID)." >&2
    exit 1
  fi
  sleep 3
done
