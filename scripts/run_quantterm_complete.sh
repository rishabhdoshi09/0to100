#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

if [[ "${1:-}" == "--restart" || "${1:-}" == "--reuse" ]]; then
  shift || true
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[COMPLETE STACK] python3 is required." >&2
  exit 1
fi

if [[ ! -d venv ]]; then
  echo "[COMPLETE STACK] Creating venv…"
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

if ! python -c 'import fastapi, uvicorn, pypdf' >/dev/null 2>&1; then
  echo "[COMPLETE STACK] Installing Python packages (first run takes a few minutes)…"
  python -m pip install --upgrade pip wheel
  python -m pip install -r requirements.txt
fi

if ! python -c 'import reportlab, fastapi, uvicorn' >/dev/null 2>&1; then
  echo "[COMPLETE STACK] Installing professional report dependencies…"
  python -m pip install 'reportlab>=4.2.0' 'fastapi>=0.115.0' 'uvicorn>=0.30.0'
fi

if [[ ! -d frontend/node_modules ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "[COMPLETE STACK] npm is required for the desk UI. Install Node.js, then re-run." >&2
    exit 1
  fi
  echo "[COMPLETE STACK] Installing frontend packages…"
  (cd frontend && npm install)
fi

if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    chmod 600 .env 2>/dev/null || true
    echo "[COMPLETE STACK] Wrote .env from .env.example. Put KITE_API_KEY and KITE_API_SECRET in it, then re-run."
    exit 2
  fi
  echo "[COMPLETE STACK] Missing .env. Create it with KITE_API_KEY and KITE_API_SECRET." >&2
  exit 2
fi

auth_rc=0
python - <<'PY' >/dev/null || auth_rc=$?
from data.kite_client import _fresh_env

if not _fresh_env("KITE_API_KEY") or not _fresh_env("KITE_API_SECRET"):
    raise SystemExit(2)
if not _fresh_env("KITE_ACCESS_TOKEN"):
    raise SystemExit(1)
try:
    from research.autonomy.auth import TOKEN_MISSING, SESSION_EXPIRED, probe_auth
    health = probe_auth()
    if health.valid:
        raise SystemExit(0)
    if health.status in {TOKEN_MISSING, SESSION_EXPIRED}:
        raise SystemExit(1)
except Exception:
    pass
raise SystemExit(0)
PY

if [[ "$auth_rc" -eq 2 ]]; then
  echo "[COMPLETE STACK] Put KITE_API_KEY and KITE_API_SECRET in .env, then run this same command again." >&2
  exit 2
fi

if [[ "$auth_rc" -eq 1 ]]; then
  if [[ "${QT_NONINTERACTIVE:-}" == "1" || ! -t 0 ]]; then
    echo "[COMPLETE STACK] Zerodha login is needed (once per trading day). Non-interactive run skipped it. Paper/EOD still work. Run: python main.py login"
  else
    echo "[COMPLETE STACK] Zerodha login is needed (once per trading day). Browser will open; paste the redirect URL here."
    python main.py login
  fi
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
  echo "[COMPLETE STACK] Starting QuantTerm terminal, market operations, autonomy and market scan…"
  bash scripts/run_quantterm.sh &
  STACK_PID=$!
}

echo "[COMPLETE STACK] One command, one terminal. Stopping any previous local desk so this run owns everything."
python scripts/local_stack.py stop --ports 5173,8765,8766 || true
sleep 1 || true

if url_ok "http://127.0.0.1:8766/health"; then
  REPORT_EXTERNAL=1
  echo "[COMPLETE STACK] Reusing research-report API at http://127.0.0.1:8766"
elif port_open 8766; then
  echo "[COMPLETE STACK] Port 8766 is occupied but /health is not ready yet; waiting." >&2
else
  start_report || true
fi

start_stack

echo "[COMPLETE STACK] Running in this terminal: desk http://127.0.0.1:5173  · API :8765  · reports :8766  · autonomy  · market scan"
echo "[COMPLETE STACK] Leave this terminal open. Ctrl-C stops everything. Do not start a second terminal."

HOME_OPENED=0
wait_for_desk() {
  # Inner stack waits for API health, then starts Vite. Do not spend the
  # whole Home budget on :5173 while :8765 is still coming up.
  local i=0
  while (( i < 90 )); do
    if url_ok "http://127.0.0.1:8765/api/health"; then
      break
    fi
    sleep 0.5 || true
    i=$((i + 1))
  done
  i=0
  while (( i < 120 )); do
    if url_ok "http://127.0.0.1:5173/" && url_ok "http://127.0.0.1:8765/api/health"; then
      return 0
    fi
    sleep 0.5 || true
    i=$((i + 1))
  done
  return 1
}

if wait_for_desk; then
  python - <<'PY' || true
from product.startup_check import print_startup_summary
raise SystemExit(print_startup_summary())
PY
  if [[ "${QT_NONINTERACTIVE:-}" != "1" && "${QT_NO_BROWSER:-}" != "1" && -t 0 && "$HOME_OPENED" != "1" ]]; then
    python - <<'PY' || true
from product.startup_check import maybe_open_home_browser
maybe_open_home_browser()
PY
    HOME_OPENED=1
  fi
else
  echo "[COMPLETE STACK] Home is still starting. Open http://127.0.0.1:5173 when the desk is up."
fi

while [[ "$STOP" != "1" ]]; do
  if [[ "$REPORT_EXTERNAL" != "1" ]]; then
    if [[ -z "${REPORT_PID:-}" ]] || ! alive "$REPORT_PID"; then
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
