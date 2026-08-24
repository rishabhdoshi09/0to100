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

REPORT_PID=""
STACK_PID=""
REPORT_EXTERNAL=0

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
  if [[ "$REPORT_EXTERNAL" == "1" ]]; then
    echo "[COMPLETE STACK] Existing report API on :8766 was left running."
  fi
  echo "[COMPLETE STACK] Stopped."
}
trap cleanup EXIT INT TERM

if url_ok "http://127.0.0.1:8766/health"; then
  REPORT_EXTERNAL=1
  echo "[COMPLETE STACK] Reusing research-report API at http://127.0.0.1:8766"
elif port_open 8766; then
  echo "[COMPLETE STACK] Port 8766 is occupied but /health is not ready." >&2
  exit 1
else
  echo "[COMPLETE STACK] Starting research-report API at http://127.0.0.1:8766 …"
  python -u -m uvicorn report_api:app --host 127.0.0.1 --port 8766 &
  REPORT_PID=$!
  sleep 1
  if ! kill -0 "$REPORT_PID" >/dev/null 2>&1; then
    echo "[COMPLETE STACK] Research-report API failed to start. Review the error above." >&2
    exit 1
  fi
fi

echo "[COMPLETE STACK] Starting QuantTerm terminal, market operations and autonomy…"
bash scripts/run_quantterm.sh &
STACK_PID=$!

while true; do
  if [[ -n "$REPORT_PID" ]] && ! kill -0 "$REPORT_PID" >/dev/null 2>&1; then
    echo "[COMPLETE STACK] REPORT API exited unexpectedly (pid=$REPORT_PID)." >&2
    exit 1
  fi
  if ! kill -0 "$STACK_PID" >/dev/null 2>&1; then
    echo "[COMPLETE STACK] QuantTerm stack exited unexpectedly (pid=$STACK_PID)." >&2
    exit 1
  fi
  sleep 3
done
