#!/usr/bin/env bash
# Install/update QuantTerm through the one canonical installed-host path.
#
# This compatibility entrypoint prepares Python/npm on macOS, verifies the
# external APFS runtime, removes historical competing launchd agents, and then
# delegates service ownership to product.host_install -> product.host_entrypoint
# -> product.host_supervisor.  It never creates a second QuantTerm supervisor.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEM_PYTHON="${QT_SYSTEM_PYTHON:-$(command -v python3)}"

EXTERNAL_VOLUME="${QT_STORAGE_EXTERNAL_VOLUME:-/Volumes/Expansion}"
STORAGE_BUNDLE="${QT_STORAGE_BUNDLE:-$EXTERNAL_VOLUME/QuantTermStorage.sparsebundle}"
STORAGE_MOUNT="${QT_STORAGE_MOUNT:-/Volumes/QuantTermStorage}"
STORAGE_RUNTIME="${QT_STORAGE_RUNTIME:-$STORAGE_MOUNT/QuantTerm/runtime}"
RUNTIME_LINK="${QT_RUNTIME_LINK:-$HOME/Library/Application Support/QuantTerm/runtime}"

# Fail closed before dependency or launchd mutation. The preflight may attach
# the already-configured sparsebundle, but it never creates/repairs a missing
# canonical runtime path and therefore cannot create split durable state.
QT_STORAGE_PREFLIGHT_REQUIRED=1 \
QT_STORAGE_EXTERNAL_VOLUME="$EXTERNAL_VOLUME" \
QT_STORAGE_BUNDLE="$STORAGE_BUNDLE" \
QT_STORAGE_MOUNT="$STORAGE_MOUNT" \
QT_STORAGE_RUNTIME="$STORAGE_RUNTIME" \
QT_RUNTIME_LINK="$RUNTIME_LINK" \
  bash "$APP_DIR/scripts/quantterm_storage_preflight.sh"
echo "[MAC SETUP] Stage 1/5: storage preflight complete"

# Stop the canonical host and PROVE launchd no longer owns it before mutating
# the shared Python environment. A best-effort bootout is not enough: the old
# implementation could continue with an exact-SHA host still running.
PYTHONPATH="$APP_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$SYSTEM_PYTHON" -m product.launchd_control stop --label com.quantterm.desk

# Historical one-off agents are not canonical owners. They remain best-effort
# cleanup after the canonical service has been proved absent.
UID_VALUE="$(id -u)"
for label in com.quantterm.ui com.quantterm.app com.quantterm.autonomy; do
  launchctl bootout "gui/$UID_VALUE/$label" 2>/dev/null || true
done

AGENTS="$HOME/Library/LaunchAgents"
for legacy in \
  "$AGENTS/com.quantterm.ui.plist" \
  "$AGENTS/com.quantterm.app.plist" \
  "$AGENTS/com.quantterm.autonomy.plist"
do
  if [[ -e "$legacy" ]]; then
    launchctl bootout "gui/$UID_VALUE" "$legacy" 2>/dev/null || launchctl unload "$legacy" 2>/dev/null || true
    rm -f "$legacy"
  fi
done

echo "[MAC SETUP] Stage 2/5: Python environment"
[ -d "$APP_DIR/venv" ] || "$SYSTEM_PYTHON" -m venv "$APP_DIR/venv"
PYTHON_BIN="${QT_PYTHON:-$APP_DIR/venv/bin/python}"

echo "[MAC SETUP] Stage 3/5: dependency sync"
run_with_deadline() {
  local seconds="$1"; shift
  "$SYSTEM_PYTHON" - "$seconds" "$@" <<'PY'
import os
import signal
import subprocess
import sys

timeout_s = float(sys.argv[1])
cmd = sys.argv[2:]
proc = subprocess.Popen(cmd, start_new_session=True)
try:
    raise SystemExit(proc.wait(timeout=timeout_s))
except subprocess.TimeoutExpired:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
    print(
        f"[MAC SETUP] ERROR: command exceeded {timeout_s:.0f}s: {' '.join(cmd)}",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(124)
PY
}
run_with_deadline 300 "$PYTHON_BIN" -m pip install --disable-pip-version-check --upgrade pip wheel
run_with_deadline 600 "$PYTHON_BIN" -m pip install --disable-pip-version-check --retries 1 --timeout 30 -r "$APP_DIR/requirements.txt"
echo "[MAC SETUP] Dependency sync complete"

echo "[MAC SETUP] Stage 4/5: secure host configuration"
[ -f "$APP_DIR/.env" ] || { cp "$APP_DIR/.env.example" "$APP_DIR/.env" 2>/dev/null || touch "$APP_DIR/.env"; }
chmod 600 "$APP_DIR/.env"

# Resolve npm while an interactive shell is available.  The absolute executable
# is persisted into the secure host env file; product.host_entrypoint validates
# it again and prepends its directory to PATH before constructing child specs.
NPM_BIN="${QT_NPM_BIN:-$(command -v npm 2>/dev/null || true)}"
if [[ -z "$NPM_BIN" || ! -x "$NPM_BIN" ]]; then
  for candidate in \
    "$HOME"/.nvm/versions/node/*/bin/npm \
    "$HOME"/.volta/bin/npm \
    "$HOME"/.asdf/shims/npm \
    /opt/homebrew/bin/npm \
    /usr/local/bin/npm
  do
    if [[ -x "$candidate" ]]; then
      NPM_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "$NPM_BIN" || ! -x "$NPM_BIN" ]]; then
  echo "QuantTerm setup requires npm for the desk UI, but npm could not be resolved." >&2
  echo "Install/activate Node.js, confirm 'command -v npm' works, then re-run setup." >&2
  exit 1
fi
NPM_BIN="$(cd "$(dirname "$NPM_BIN")" && pwd -P)/$(basename "$NPM_BIN")"

# Persist every value required to re-establish the APFS runtime after login,
# reboot, or a removable-disk disconnect. host_entrypoint reads this secure file
# before touching strict runtime paths and invokes the same fail-closed preflight
# used above. Values are replaced atomically by key; unrelated operator secrets
# and configuration remain untouched.
"$PYTHON_BIN" - "$APP_DIR/.env" \
  "QT_NPM_BIN=$NPM_BIN" \
  "QT_STORAGE_PREFLIGHT_REQUIRED=1" \
  "QT_STORAGE_EXTERNAL_VOLUME=$EXTERNAL_VOLUME" \
  "QT_STORAGE_BUNDLE=$STORAGE_BUNDLE" \
  "QT_STORAGE_MOUNT=$STORAGE_MOUNT" \
  "QT_STORAGE_RUNTIME=$STORAGE_RUNTIME" \
  "QT_RUNTIME_LINK=$RUNTIME_LINK" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
updates = dict(item.split("=", 1) for item in sys.argv[2:])
rows = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
kept = []
for row in rows:
    stripped = row.strip()
    candidate = stripped[7:].lstrip() if stripped.startswith("export ") else stripped
    key = candidate.split("=", 1)[0].strip() if "=" in candidate else ""
    if key in updates:
        continue
    kept.append(row)
kept.extend(f"{key}={value}" for key, value in updates.items())
path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
PY
chmod 600 "$APP_DIR/.env"

export PYTHON="$PYTHON_BIN"
export QT_NPM_BIN="$NPM_BIN"
# The macOS runtime is already adopted and verified by preflight. The strict
# installer adapter must never mkdir this path if the removable volume vanishes
# during the update window; it fails closed instead.
export QT_RUNTIME_ROOT_REQUIRE_EXISTING=1
echo "[MAC SETUP] Stage 5/5: install/re-pin canonical launchd host to current checkout"
exec "$APP_DIR/scripts/install_quantterm_host.sh" \
  --runtime-root "$STORAGE_RUNTIME" \
  --env-file "$APP_DIR/.env" \
  --manager launchd \
  "$@"
