#!/usr/bin/env bash
# Install/update QuantTerm as one canonical launchd agent on the current Mac.
# Canonical chain: setup_mac.sh -> run_quantterm_mac.sh -> run_quantterm_complete.sh.
# The complete launcher owns autonomy, market_ops, APIs and the desk; a second
# autonomy LaunchAgent would create competing ownership/restart paths.
# Legacy com.quantterm.autonomy used <string>autonomy</string>; setup removes it.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEM_PYTHON="${QT_SYSTEM_PYTHON:-$(command -v python3)}"

EXTERNAL_VOLUME="${QT_STORAGE_EXTERNAL_VOLUME:-/Volumes/Expansion}"
STORAGE_BUNDLE="${QT_STORAGE_BUNDLE:-$EXTERNAL_VOLUME/QuantTermStorage.sparsebundle}"
STORAGE_MOUNT="${QT_STORAGE_MOUNT:-/Volumes/QuantTermStorage}"
STORAGE_RUNTIME="${QT_STORAGE_RUNTIME:-$STORAGE_MOUNT/QuantTerm/runtime}"
RUNTIME_LINK="${QT_RUNTIME_LINK:-$HOME/Library/Application Support/QuantTerm/runtime}"

# Refuse installation against an absent/wrong runtime. This is intentionally
# before dependency installation or launchd mutation so setup cannot create a
# half-installed service while the durable store is unavailable.
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
chmod 600 "$APP_DIR/.env" 2>/dev/null || true

# Resolve npm while we are still in the user's interactive shell.  launchd does
# not source shell profiles, so nvm/Volta/asdf installs can otherwise disappear
# from PATH at reboot even though `npm` works in Terminal.
NPM_BIN="${QT_NPM_BIN:-$(command -v npm 2>/dev/null || true)}"
if [[ -z "$NPM_BIN" || ! -x "$NPM_BIN" ]]; then
  for candidate in \
    "$HOME"/.nvm/versions/node/*/bin/npm \
    "$HOME"/.volta/bin/npm \
    "$HOME"/.asdf/shims/npm
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
NPM_BIN_DIR="$(cd "$(dirname "$NPM_BIN")" && pwd -P)"
LAUNCH_PATH="$NPM_BIN_DIR:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

AGENTS="$HOME/Library/LaunchAgents"
APP_PLIST="$AGENTS/com.quantterm.ui.plist"
OLD_AUTO_PLIST="$AGENTS/com.quantterm.autonomy.plist"
OLD_COMBINED_PLIST="$AGENTS/com.quantterm.app.plist"
LAUNCH_LOG_DIR="$HOME/Library/Logs/QuantTerm"
mkdir -p "$AGENTS" "$LAUNCH_LOG_DIR"

sudo pmset -a sleep 0 displaysleep 10 || true

cat > "$APP_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.quantterm.ui</string>
<key>ProgramArguments</key><array>
<string>/bin/bash</string>
<string>$APP_DIR/scripts/run_quantterm_mac.sh</string>
</array>
<key>WorkingDirectory</key><string>$APP_DIR</string>
<key>EnvironmentVariables</key><dict>
<key>TZ</key><string>Asia/Kolkata</string>
<key>PYTHONPATH</key><string>$APP_DIR</string>
<key>QT_NONINTERACTIVE</key><string>1</string>
<key>QT_STORAGE_PREFLIGHT_REQUIRED</key><string>1</string>
<key>QT_STORAGE_EXTERNAL_VOLUME</key><string>$EXTERNAL_VOLUME</string>
<key>QT_STORAGE_BUNDLE</key><string>$STORAGE_BUNDLE</string>
<key>QT_STORAGE_MOUNT</key><string>$STORAGE_MOUNT</string>
<key>QT_STORAGE_RUNTIME</key><string>$STORAGE_RUNTIME</string>
<key>QT_RUNTIME_LINK</key><string>$RUNTIME_LINK</string>
<key>QT_RUNTIME_ROOT</key><string>$RUNTIME_LINK</string>
<key>QT_NPM_BIN</key><string>$NPM_BIN</string>
<key>PATH</key><string>$LAUNCH_PATH</string>
</dict>
<key>RunAtLoad</key><true/>
<key>KeepAlive</key><dict>
  <key>PathState</key><dict>
    <key>$EXTERNAL_VOLUME</key><true/>
  </dict>
</dict>
<key>ThrottleInterval</key><integer>60</integer>
<key>StandardOutPath</key><string>$LAUNCH_LOG_DIR/launchd.log</string>
<key>StandardErrorPath</key><string>$LAUNCH_LOG_DIR/launchd.log</string>
</dict></plist>
PLIST

plutil -lint "$APP_PLIST" >/dev/null

# Remove historical competing service definitions before loading the one owner.
for old in "$OLD_COMBINED_PLIST" "$OLD_AUTO_PLIST"; do
  launchctl bootout "gui/$(id -u)" "$old" 2>/dev/null || launchctl unload "$old" 2>/dev/null || true
  rm -f "$old"
done

launchctl bootout "gui/$(id -u)" "$APP_PLIST" 2>/dev/null || launchctl unload "$APP_PLIST" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$APP_PLIST" 2>/dev/null || launchctl load -w "$APP_PLIST"
launchctl kickstart -k "gui/$(id -u)/com.quantterm.ui" || true

echo "QuantTerm canonical macOS agent installed."
echo "External runtime: $STORAGE_RUNTIME"
echo "npm: $NPM_BIN"
echo "Daily login: cd '$APP_DIR' && '$PYTHON_BIN' main.py login"
echo "Desk: http://127.0.0.1:5173"
echo "Launch log: $LAUNCH_LOG_DIR/launchd.log"
