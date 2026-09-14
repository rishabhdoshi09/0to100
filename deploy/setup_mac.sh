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

[ -d "$APP_DIR/venv" ] || "$SYSTEM_PYTHON" -m venv "$APP_DIR/venv"
PYTHON_BIN="${QT_PYTHON:-$APP_DIR/venv/bin/python}"
"$PYTHON_BIN" -m pip install --upgrade pip wheel
"$PYTHON_BIN" -m pip install -r "$APP_DIR/requirements.txt"

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

"$PYTHON_BIN" - "$APP_DIR/.env" "$NPM_BIN" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
npm = sys.argv[2]
rows = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
kept = []
for row in rows:
    stripped = row.strip()
    candidate = stripped[7:].lstrip() if stripped.startswith("export ") else stripped
    if candidate.startswith("QT_NPM_BIN="):
        continue
    kept.append(row)
kept.append(f"QT_NPM_BIN={npm}")
path.write_text("\n".join(kept).rstrip() + "\n", encoding="utf-8")
PY
chmod 600 "$APP_DIR/.env"

# Remove every historical macOS QuantTerm owner before the canonical installer
# loads com.quantterm.desk.  This makes setup idempotent and prevents two stacks
# from racing for ports, autonomy ownership, or the same durable runtime.
AGENTS="$HOME/Library/LaunchAgents"
for legacy in \
  "$AGENTS/com.quantterm.ui.plist" \
  "$AGENTS/com.quantterm.app.plist" \
  "$AGENTS/com.quantterm.autonomy.plist"
do
  if [[ -e "$legacy" ]]; then
    launchctl bootout "gui/$(id -u)" "$legacy" 2>/dev/null || launchctl unload "$legacy" 2>/dev/null || true
    rm -f "$legacy"
  fi
done

export PYTHON="$PYTHON_BIN"
export QT_NPM_BIN="$NPM_BIN"
exec "$APP_DIR/scripts/install_quantterm_host.sh" \
  --runtime-root "$STORAGE_RUNTIME" \
  --env-file "$APP_DIR/.env" \
  --manager launchd \
  "$@"
