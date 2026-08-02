#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck source=stack_lib.sh
source "$ROOT/scripts/stack_lib.sh"

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
API_EXTERNAL=0

cleanup() {
  echo
  echo "[STACK] Stopping QuantTerm child services…"
  if [[ -n "$FRONTEND_PID" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$API_PID" ]]; then
    kill "$API_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "$AUTONOMY_PID" ]]; then
    kill "$AUTONOMY_PID" >/dev/null 2>&1 || true
  fi
  wait >/dev/null 2>&1 || true
  # Vite/npm often outlive the launcher PID; reap by bind port.
  stack_free_port 5173 "Vite dev server"
  if [[ "$API_EXTERNAL" != "1" ]]; then
    stack_free_port 8765 "Terminal API"
  fi
  if [[ "$AUTONOMY_EXTERNAL" == "1" ]]; then
    echo "[STACK] Existing external autonomy supervisor was left running."
  fi
  if [[ "$API_EXTERNAL" == "1" ]]; then
    echo "[STACK] External terminal API on :8765 was left running."
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

API_HEALTH="http://127.0.0.1:8765/api/health"
set +e
# Direct call — do not capture via $(...); that kills the backgrounded uvicorn.
stack_start_or_reuse_uvicorn 8765 "terminal_product_api:app" "$API_HEALTH" "Terminal API"
api_rc=$STACK_UVICORN_RC
API_PID=$STACK_UVICORN_PID
set -e

if [[ "$api_rc" == 2 ]]; then
  API_EXTERNAL=1
  API_PID=""
else
  API_EXTERNAL=0
  if [[ -z "$API_PID" ]]; then
    echo "[STACK] Terminal API failed to start (no pid)." >&2
    exit 1
  fi
fi

echo "[STACK] Waiting for terminal API (bhav load + market ops bootstrap can take ~15–45s)…"
if [[ "$API_EXTERNAL" == "1" ]]; then
  stack_wait_for_health "$API_HEALTH" "" "Terminal API" || exit 1
else
  stack_wait_for_health "$API_HEALTH" "$API_PID" "Terminal API" || exit 1
fi
echo "[STACK] Terminal API ready."

stack_free_port 5173 "Vite dev server"

echo "[STACK] Starting dedicated terminal at http://127.0.0.1:5173 …"
# Avoid `( cd …; npm … ) &` — killing that subshell PID leaves an orphaned Vite
# that keeps proxying to a dead :8765 and surfaces ECONNREFUSED in the browser.
npm --prefix "$ROOT/frontend" run dev -- --host 127.0.0.1 --port 5173 &
FRONTEND_PID=$!

# Confirm Vite bound before advertising ready (API is already healthy above).
vite_bound=0
for _ in $(seq 1 60); do
  if ! kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    echo "[STACK] Vite exited during startup. Review errors above." >&2
    exit 1
  fi
  if [[ -n "$(stack_pids_on_port 5173)" ]]; then
    vite_bound=1
    break
  fi
  sleep 0.5
done
if [[ "$vite_bound" != "1" ]]; then
  echo "[STACK] Vite did not bind :5173 within 30s." >&2
  exit 1
fi

echo "[STACK] QuantTerm is ready. Open http://127.0.0.1:5173"
echo "[STACK] Keep this terminal open; Ctrl-C stops services started by this script."
echo "[STACK] If the UI still errors, run: bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_complete.sh"

while true; do
  if ! kill -0 "$FRONTEND_PID" >/dev/null 2>&1; then
    echo "[STACK] FRONTEND process exited unexpectedly (pid=$FRONTEND_PID)."
    exit 1
  fi
  if [[ "$API_EXTERNAL" != "1" ]] && [[ -n "$API_PID" ]] && ! kill -0 "$API_PID" >/dev/null 2>&1; then
    echo "[STACK] API process exited unexpectedly (pid=$API_PID)."
    echo "[STACK] Tip: bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_complete.sh" >&2
    exit 1
  fi
  if ! stack_health_ok "$API_HEALTH"; then
    echo "[STACK] Terminal API health check failed at $API_HEALTH — backend may have stopped." >&2
    echo "[STACK] Vite proxies /api → :8765; restart with: bash scripts/stop_quantterm.sh && bash scripts/run_quantterm_complete.sh" >&2
    exit 1
  fi
  if [[ -n "$AUTONOMY_PID" ]] && ! kill -0 "$AUTONOMY_PID" >/dev/null 2>&1; then
    echo "[STACK] AUTONOMY process exited unexpectedly (pid=$AUTONOMY_PID)."
    exit 1
  fi
  sleep 3
done
