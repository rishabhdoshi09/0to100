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

if ! python -c 'import reportlab, fastapi, uvicorn' >/dev/null 2>&1; then
  echo "[STACK] Installing professional report dependencies…"
  python -m pip install 'reportlab>=4.2.0' 'fastapi>=0.115.0' 'uvicorn>=0.30.0'
fi

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
    echo "[COMPLETE STACK] External report API on :8766 was left running."
  fi
  echo "[COMPLETE STACK] Stopped."
}
trap cleanup EXIT INT TERM

REPORT_HEALTH="http://127.0.0.1:8766/health"
set +e
# Direct call — do not capture via $(...); that kills the backgrounded uvicorn.
stack_start_or_reuse_uvicorn 8766 "report_api:app" "$REPORT_HEALTH" "Research-report API"
report_rc=$STACK_UVICORN_RC
REPORT_PID=$STACK_UVICORN_PID
set -e

if [[ "$report_rc" == 2 ]]; then
  REPORT_EXTERNAL=1
  REPORT_PID=""
else
  REPORT_EXTERNAL=0
  if [[ -z "$REPORT_PID" ]]; then
    echo "[COMPLETE STACK] Research-report API failed to start." >&2
    exit 1
  fi
  # Allow enough time for import + bind; process-death is still detected early.
  if ! stack_wait_for_health "$REPORT_HEALTH" "$REPORT_PID" "Research-report API" 60; then
    exit 1
  fi
fi

echo "[COMPLETE STACK] Starting QuantTerm terminal, market operations and autonomy…"
bash scripts/run_quantterm.sh &
STACK_PID=$!

while true; do
  if [[ "$REPORT_EXTERNAL" != "1" ]] && [[ -n "$REPORT_PID" ]] && ! kill -0 "$REPORT_PID" >/dev/null 2>&1; then
    echo "[COMPLETE STACK] REPORT API exited unexpectedly (pid=$REPORT_PID)." >&2
    exit 1
  fi
  if ! stack_health_ok "$REPORT_HEALTH"; then
    echo "[COMPLETE STACK] Report API health check failed at $REPORT_HEALTH." >&2
    exit 1
  fi
  if ! kill -0 "$STACK_PID" >/dev/null 2>&1; then
    echo "[COMPLETE STACK] QuantTerm stack exited unexpectedly (pid=$STACK_PID)." >&2
    echo "[COMPLETE STACK] Tip: bash scripts/stop_quantterm.sh  then  bash scripts/run_quantterm_complete.sh" >&2
    exit 1
  fi
  sleep 3
done
