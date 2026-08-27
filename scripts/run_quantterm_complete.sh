#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESTART=0
if [[ "${1:-}" == "--restart" ]]; then
  RESTART=1
  shift || true
fi

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

alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

REPORT_PID=""
STACK_PID=""
REPORT_EXTERNAL=0
STOP=0
CLEANED=0

cleanup() {
  if [[ "$CLEANED" == "1" ]]; then
    return
  fi
  CLEANED=1
  STOP=1
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

on_stop() {
  cleanup
  exit 0
}

trap on_stop INT TERM
trap cleanup EXIT

start_report() {
  echo "[COMPLETE STACK] Starting research-report API at http://127.0.0.1:8766 …"
  python -u -m uvicorn report_api:app --host 127.0.0.1 --port 8766 &
  REPORT_PID=$!
  sleep 1 || true
  if alive "$REPORT_PID"; then
    return 0
  fi
  echo "[COMPLETE STACK] Research-report API failed to start; will retry." >&2
  REPORT_PID=""
  return 1
}

start_stack() {
  echo "[COMPLETE STACK] Starting QuantTerm terminal, market operations and autonomy…"
  if [[ "$RESTART" == "1" ]]; then
    QT_RESTART=1 bash scripts/run_quantterm.sh --restart &
  else
    bash scripts/run_quantterm.sh &
  fi
  STACK_PID=$!
}

if [[ "$RESTART" == "1" ]]; then
  echo "[COMPLETE STACK] --restart: stopping the local desk, API, reports and autonomy so this run loads current code."
  python scripts/local_stack.py stop --ports 5173,8765,8766 || true
  sleep 1 || true
fi

if url_ok "http://127.0.0.1:8766/health"; then
  REPORT_EXTERNAL=1
  echo "[COMPLETE STACK] Reusing research-report API at http://127.0.0.1:8766"
elif port_open 8766; then
  echo "[COMPLETE STACK] Port 8766 is occupied but /health is not ready yet; waiting." >&2
else
  start_report || true
fi

start_stack

echo "[COMPLETE STACK] Running. Desk http://127.0.0.1:5173  · API :8765  · reports :8766"
echo "[COMPLETE STACK] Leave this terminal open. Ctrl-C stops the stack."
echo "[COMPLETE STACK] After git pull, re-run: bash scripts/run_quantterm_complete.sh --restart"

while [[ "$STOP" != "1" ]]; do
  if [[ "$REPORT_EXTERNAL" != "1" ]]; then
    if url_ok "http://127.0.0.1:8766/health"; then
      :
    elif ! alive "$REPORT_PID"; then
      echo "[COMPLETE STACK] Report API is down; restarting."
      start_report || true
    fi
  fi
  if ! alive "$STACK_PID"; then
    echo "[COMPLETE STACK] Inner stack script ended; restarting it. The desk is not supposed to go idle."
    start_stack
  fi
  sleep 3 || true
done
