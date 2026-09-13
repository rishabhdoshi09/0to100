#!/usr/bin/env bash
# macOS production entrypoint: verify/mount external runtime, supervise the
# canonical complete QuantTerm launcher, and stop the stack if storage vanishes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export QT_STORAGE_PREFLIGHT_REQUIRED=1
PREFLIGHT="$ROOT/scripts/quantterm_storage_preflight.sh"
WATCH_INTERVAL_S="${QT_STORAGE_WATCH_INTERVAL_S:-15}"

bash "$PREFLIGHT"

CHILD_PID=""
STOPPING=0
stop_child() {
  if [[ "$STOPPING" == "1" ]]; then
    return
  fi
  STOPPING=1
  if [[ -n "$CHILD_PID" ]] && kill -0 "$CHILD_PID" >/dev/null 2>&1; then
    kill -TERM "$CHILD_PID" >/dev/null 2>&1 || true
    wait "$CHILD_PID" 2>/dev/null || true
  fi
}
trap 'stop_child; exit 0' INT TERM

bash "$ROOT/scripts/run_quantterm_complete.sh" "$@" &
CHILD_PID=$!

while kill -0 "$CHILD_PID" >/dev/null 2>&1; do
  sleep "$WATCH_INTERVAL_S" || true
  if ! QT_STORAGE_PREFLIGHT_ATTACH=0 bash "$PREFLIGHT" >/dev/null 2>&1; then
    echo "[MAC RUNTIME] External QuantTerm storage was lost or became invalid; stopping the complete stack." >&2
    stop_child
    exit 78
  fi
done

set +e
wait "$CHILD_PID"
RC=$?
set -e
CHILD_PID=""
exit "$RC"
