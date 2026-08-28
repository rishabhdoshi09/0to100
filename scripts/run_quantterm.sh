#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

if [[ "${1:-}" == "--restart" || "${1:-}" == "--reuse" ]]; then
  shift || true
fi

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
sock = socket.socket(); sock.settimeout(0.4)
try:
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
finally:
    sock.close()
PY
}

url_ok() {
  local url="$1"
  python - "$url" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=1.5) as response:
        raise SystemExit(0 if int(response.status) == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

wait_for_api() {
  local tries="${1:-90}"; local i=0
  while (( i < tries )); do
    if url_ok "http://127.0.0.1:8765/api/health"; then return 0; fi
    sleep 0.5 || true; i=$((i + 1))
  done
  return 1
}

alive() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

market_ops_healthy() {
  python - <<'PY' >/dev/null 2>&1
import json, os, time
from pathlib import Path
p = Path("logs/market_ops/runtime.json")
try:
    r = json.loads(p.read_text(encoding="utf-8"))
    pid = int(r.get("worker_pid") or 0)
    hb = float(r.get("heartbeat_epoch") or 0)
    if not r.get("process_running") or pid <= 1 or time.time() - hb > 8:
        raise SystemExit(1)
    os.kill(pid, 0)
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

stop_stale_market_ops() {
  python - <<'PY' >/dev/null 2>&1 || true
import json, os, signal, subprocess, time
from pathlib import Path
p = Path("logs/market_ops/runtime.json")
try:
    r = json.loads(p.read_text(encoding="utf-8"))
    pid = int(r.get("worker_pid") or 0)
except Exception:
    pid = 0
if pid <= 1 or pid == os.getpid():
    raise SystemExit(0)
try:
    command = subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=0.5,
    ).strip()
except Exception:
    raise SystemExit(0)
if "operations.market_ops" not in command:
    raise SystemExit(0)
try:
    os.kill(pid, signal.SIGTERM)
except OSError:
    raise SystemExit(0)
deadline = time.time() + 1.5
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        raise SystemExit(0)
    time.sleep(0.05)
try:
    command = subprocess.check_output(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=0.5,
    ).strip()
except Exception:
    command = ""
if "operations.market_ops" in command:
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
PY
}

AUTONOMY_PID=""
MARKET_OPS_PID=""
API_PID=""
FRONTEND_PID=""
AUTONOMY_EXTERNAL=0
MARKET_OPS_EXTERNAL=0
API_EXTERNAL=0
FRONTEND_EXTERNAL=0
SCAN_KICKED=0
STOP=0
CLEANED=0

cleanup() {
  if [[ "$CLEANED" == "1" ]]; then return; fi
  CLEANED=1; STOP=1
  echo
  echo "[STACK] Stopping QuantTerm child services…"
  for pid in "$FRONTEND_PID" "$API_PID" "$MARKET_OPS_PID" "$AUTONOMY_PID"; do
    if [[ -n "$pid" ]]; then kill "$pid" >/dev/null 2>&1 || true; fi
  done
  wait >/dev/null 2>&1 || true
  [[ "$AUTONOMY_EXTERNAL" == "1" ]] && echo "[STACK] Existing external autonomy supervisor was left running."
  [[ "$MARKET_OPS_EXTERNAL" == "1" ]] && echo "[STACK] Existing external market-operations worker was left running."
  [[ "$API_EXTERNAL" == "1" ]] && echo "[STACK] Existing market API on :8765 was left running."
  [[ "$FRONTEND_EXTERNAL" == "1" ]] && echo "[STACK] Existing RecoWealth desk on :5173 was left running."
  echo "[STACK] Child services stopped."
}

on_stop() { cleanup; exit 0; }
trap on_stop INT TERM
trap cleanup EXIT

start_autonomy() {
  echo "[STACK] Starting autonomy supervisor…"
  python -u main.py autonomy &
  AUTONOMY_PID=$!
  sleep 1 || true
  if alive "$AUTONOMY_PID"; then return 0; fi
  if python - <<'PY' >/dev/null 2>&1
from product.autonomy_status import read_autonomy_status
raise SystemExit(0 if read_autonomy_status().get("running") else 1)
PY
  then
    AUTONOMY_PID=""; AUTONOMY_EXTERNAL=1
    echo "[STACK] Another healthy supervisor acquired the lock; reusing it."
    return 0
  fi
  echo "[STACK] Autonomy failed to stay alive; will retry." >&2
  AUTONOMY_PID=""; return 1
}

start_market_ops() {
  echo "[STACK] Starting market-operations worker (scan/news/long-term/data lanes)…"
  python -u -m operations.market_ops &
  MARKET_OPS_PID=$!
  local i=0
  while (( i < 30 )); do
    if market_ops_healthy; then
      echo "[STACK] Market operations READY · pid=${MARKET_OPS_PID}"
      return 0
    fi
    if ! alive "$MARKET_OPS_PID"; then break; fi
    sleep 0.1 || true; i=$((i + 1))
  done
  if market_ops_healthy; then
    MARKET_OPS_PID=""; MARKET_OPS_EXTERNAL=1
    echo "[STACK] Another healthy market-operations worker owns the lock; reusing it."
    return 0
  fi
  echo "[STACK] Market operations failed to become healthy; will retry." >&2
  MARKET_OPS_PID=""; return 1
}

start_api() {
  echo "[STACK] Starting local API at http://127.0.0.1:8765 …"
  # terminal_product_api_parallel imports the canonical terminal_product_api:app
  # and only corrects performance-safe operation routing.
  python -u -m uvicorn terminal_product_api_parallel:app --host 127.0.0.1 --port 8765 &
  API_PID=$!
  sleep 0.5 || true
  if ! alive "$API_PID"; then
    echo "[STACK] Market API exited before becoming healthy; will retry." >&2
    API_PID=""; return 1
  fi
  return 0
}

start_frontend() {
  echo "[STACK] Starting dedicated terminal at http://127.0.0.1:5173 …"
  (cd frontend && npm run dev -- --host 127.0.0.1) &
  FRONTEND_PID=$!
}

kick_scan() {
  if [[ "$SCAN_KICKED" == "1" ]]; then return 0; fi
  if ! market_ops_healthy; then
    echo "[STACK] Scan kick waiting for market-operations worker…" >&2
    return 1
  fi
  if ! url_ok "http://127.0.0.1:8765/api/health"; then return 1; fi
  echo "[STACK] Queueing market scan, news and long-term funds in this terminal…"
  if python scripts/local_stack.py scan; then SCAN_KICKED=1; return 0; fi
  return 1
}

echo "[STACK] Stopping any previous API, desk, autonomy and market operations so this terminal owns them."
python scripts/local_stack.py stop --ports 5173,8765 || true
sleep 1 || true

if python - <<'PY' >/dev/null 2>&1
from product.autonomy_status import read_autonomy_status
raise SystemExit(0 if read_autonomy_status().get("running") else 1)
PY
then
  AUTONOMY_EXTERNAL=1
  echo "[STACK] A healthy autonomy supervisor is already running; reusing it."
else
  start_autonomy || true
fi

if market_ops_healthy; then
  MARKET_OPS_EXTERNAL=1
  echo "[STACK] A healthy market-operations worker is already running; reusing it."
else
  stop_stale_market_ops
  start_market_ops || true
fi

if url_ok "http://127.0.0.1:8765/api/health"; then
  API_EXTERNAL=1
  echo "[STACK] Reusing market API at http://127.0.0.1:8765"
elif port_open 8765; then
  echo "[STACK] Port 8765 is occupied but /api/health is not ready yet; waiting." >&2
else
  start_api || true
fi

if port_open 5173; then
  FRONTEND_EXTERNAL=1
  echo "[STACK] Reusing dedicated terminal at http://127.0.0.1:5173"
else
  start_frontend
fi

kick_scan || true

echo "[STACK] QuantTerm is running in this terminal: desk :5173, API :8765, autonomy, market operations, market scan."
echo "[STACK] Ctrl-C is the stop signal. A child crash is restarted; it does not stop the desk."

while [[ "$STOP" != "1" ]]; do
  if [[ "$MARKET_OPS_EXTERNAL" != "1" ]]; then
    if [[ -z "${MARKET_OPS_PID:-}" ]] || ! alive "$MARKET_OPS_PID" || ! market_ops_healthy; then
      if market_ops_healthy; then
        MARKET_OPS_EXTERNAL=1; MARKET_OPS_PID=""
      else
        echo "[STACK] Market operations is down/stale; restarting."
        stop_stale_market_ops
        start_market_ops || true
        SCAN_KICKED=0
      fi
    fi
  elif ! market_ops_healthy; then
    MARKET_OPS_EXTERNAL=0
    echo "[STACK] Reused market-operations worker became stale; taking ownership."
    stop_stale_market_ops
    start_market_ops || true
    SCAN_KICKED=0
  fi

  if [[ "$API_EXTERNAL" != "1" ]]; then
    if [[ -z "${API_PID:-}" ]] || ! alive "$API_PID"; then
      echo "[STACK] Market API is down; restarting."
      start_api || true
    fi
  fi
  if [[ "$FRONTEND_EXTERNAL" != "1" ]] && ! alive "$FRONTEND_PID"; then
    if port_open 5173; then
      FRONTEND_EXTERNAL=1; FRONTEND_PID=""
      echo "[STACK] RecoWealth desk is already on :5173; reusing it."
    else
      echo "[STACK] RecoWealth desk is down; restarting."
      start_frontend
    fi
  fi
  if [[ "$AUTONOMY_EXTERNAL" != "1" ]] && ! alive "$AUTONOMY_PID"; then
    if python - <<'PY' >/dev/null 2>&1
from product.autonomy_status import read_autonomy_status
raise SystemExit(0 if read_autonomy_status().get("running") else 1)
PY
    then
      AUTONOMY_EXTERNAL=1; AUTONOMY_PID=""
    else
      echo "[STACK] Autonomy is down; restarting."
      start_autonomy || true
    fi
  fi
  if [[ "$SCAN_KICKED" != "1" ]]; then kick_scan || true; fi
  sleep 1 || true
done
