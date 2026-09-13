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

# macOS Catalina does not ship the util-linux `setsid` command used by the
# generic Unix launchers to give Vite its own process group.  Export a Bash
# compatibility function only when the native command is absent.  The child
# complete/inner Bash launchers inherit it, and the backgrounded function
# `exec`s Python so $! remains the session leader PID.  Python then replaces
# itself with npm after os.setsid(), preserving the existing group-cleanup
# contract without requiring Homebrew or another machine dependency.
if ! command -v setsid >/dev/null 2>&1; then
  setsid() {
    exec python - "$@" <<'PY'
import os
import sys

if len(sys.argv) < 2:
    raise SystemExit("setsid compatibility launcher requires a command")
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])
PY
  }
  export -f setsid
  echo "[MAC RUNTIME] Native setsid unavailable; using Python session launcher."
fi

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
