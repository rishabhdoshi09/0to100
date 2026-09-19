#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

# Shutdown is deliberately owner-directed. Do not enumerate/kill whatever happens
# to listen on QuantTerm's ports: another checkout or an unrelated local service
# may own them. The complete-stack supervisor already owns bounded child cleanup.
OWNER_JSON="$(python3 scripts/local_stack.py owner-status 2>/dev/null || echo '{}')"

read -r OWNER_PID OWNER_ROOT < <(python3 - "$OWNER_JSON" <<'PY'
import json, sys
try:
    payload = json.loads(sys.argv[1] or "{}")
except Exception:
    payload = {}
try:
    pid = int(payload.get("pid") or 0)
except (TypeError, ValueError):
    pid = 0
root = str(payload.get("root") or "")
print(pid, root)
PY
)

if [[ "$OWNER_PID" -le 1 ]]; then
  echo "[QUANTTERM STOP] No recorded complete-stack owner. Nothing was killed."
  exit 0
fi

if ! kill -0 "$OWNER_PID" >/dev/null 2>&1; then
  echo "[QUANTTERM STOP] Recorded owner pid $OWNER_PID is no longer alive. Nothing was killed."
  exit 0
fi

# Verify the recorded PID still looks like the QuantTerm complete-stack launcher.
# PID reuse must fail closed rather than terminating an unrelated process.
COMMAND="$(ps -p "$OWNER_PID" -o command= 2>/dev/null || true)"
if [[ "$COMMAND" != *"run_quantterm_complete.sh"* ]]; then
  echo "[QUANTTERM STOP] Refusing to signal pid $OWNER_PID: it is not the complete-stack launcher."
  exit 1
fi

if [[ -n "$OWNER_ROOT" && "$OWNER_ROOT" != "$ROOT" ]]; then
  echo "[QUANTTERM STOP] QuantTerm is owned by another checkout: $OWNER_ROOT"
  echo "[QUANTTERM STOP] Signalling its recorded supervisor only; this checkout will not enumerate or kill ports."
fi

kill -TERM "$OWNER_PID"

DEADLINE=$((SECONDS + ${QT_STOP_TIMEOUT_S:-25}))
while kill -0 "$OWNER_PID" >/dev/null 2>&1; do
  if (( SECONDS >= DEADLINE )); then
    echo "[QUANTTERM STOP] Supervisor did not finish bounded cleanup in time. Refusing SIGKILL; inspect its terminal/logs." >&2
    exit 1
  fi
  sleep 0.2
done

echo "[QUANTTERM STOP] QuantTerm complete-stack supervisor stopped cleanly."
