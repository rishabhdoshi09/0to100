#!/usr/bin/env bash
# Install/update QuantTerm as two launchd agents on the current Mac checkout.
set -euo pipefail
APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEM_PYTHON="${QT_SYSTEM_PYTHON:-$(command -v python3)}"
[ -d "$APP_DIR/venv" ] || "$SYSTEM_PYTHON" -m venv "$APP_DIR/venv"
PYTHON_BIN="${QT_PYTHON:-$APP_DIR/venv/bin/python}"
"$PYTHON_BIN" -m pip install --upgrade pip wheel
"$PYTHON_BIN" -m pip install -r "$APP_DIR/requirements.txt"
[ -f "$APP_DIR/.env" ] || { cp "$APP_DIR/.env.example" "$APP_DIR/.env" 2>/dev/null || touch "$APP_DIR/.env"; }
chmod 600 "$APP_DIR/.env" 2>/dev/null || true
AGENTS="$HOME/Library/LaunchAgents"
UI_PLIST="$AGENTS/com.quantterm.ui.plist"
AUTO_PLIST="$AGENTS/com.quantterm.autonomy.plist"
mkdir -p "$AGENTS" "$APP_DIR/logs/autonomy" "$APP_DIR/logs/intelligence" \
         "$APP_DIR/logs/snapshots" "$APP_DIR/logs/kite_history" "$APP_DIR/logs/product"

sudo pmset -a sleep 0 displaysleep 10 || true

cat > "$UI_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.quantterm.ui</string>
<key>ProgramArguments</key><array>
<string>$PYTHON_BIN</string><string>-m</string><string>streamlit</string><string>run</string>
<string>$APP_DIR/app.py</string><string>--server.port</string><string>8501</string>
<string>--server.headless</string><string>true</string>
</array>
<key>WorkingDirectory</key><string>$APP_DIR</string>
<key>EnvironmentVariables</key><dict><key>TZ</key><string>Asia/Kolkata</string></dict>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ThrottleInterval</key><integer>10</integer>
<key>StandardOutPath</key><string>$APP_DIR/logs/ui.log</string>
<key>StandardErrorPath</key><string>$APP_DIR/logs/ui.log</string>
</dict></plist>
PLIST

cat > "$AUTO_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>com.quantterm.autonomy</string>
<key>ProgramArguments</key><array>
<string>$PYTHON_BIN</string><string>$APP_DIR/main.py</string><string>autonomy</string>
<string>--interval</string><string>15</string>
</array>
<key>WorkingDirectory</key><string>$APP_DIR</string>
<key>EnvironmentVariables</key><dict>
<key>TZ</key><string>Asia/Kolkata</string><key>QT_AUTONOMY_OWNER</key><string>1</string>
</dict>
<key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ThrottleInterval</key><integer>10</integer>
<key>StandardOutPath</key><string>$APP_DIR/logs/autonomy.log</string>
<key>StandardErrorPath</key><string>$APP_DIR/logs/autonomy.log</string>
</dict></plist>
PLIST

# Remove the obsolete combined agent, then idempotently reload both real agents.
OLD="$AGENTS/com.quantterm.app.plist"
launchctl bootout "gui/$(id -u)" "$OLD" 2>/dev/null || launchctl unload "$OLD" 2>/dev/null || true
rm -f "$OLD"
for plist in "$UI_PLIST" "$AUTO_PLIST"; do
  launchctl bootout "gui/$(id -u)" "$plist" 2>/dev/null || launchctl unload "$plist" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$plist" 2>/dev/null || launchctl load -w "$plist"
done
launchctl kickstart -k "gui/$(id -u)/com.quantterm.autonomy" || true
launchctl kickstart -k "gui/$(id -u)/com.quantterm.ui" || true

echo "QuantTerm UI + autonomy agents installed."
echo "Daily login: cd '$APP_DIR' && '$PYTHON_BIN' main.py login"
echo "UI log: $APP_DIR/logs/ui.log"
echo "Autonomy log: $APP_DIR/logs/autonomy.log"
