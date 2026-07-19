#!/usr/bin/env bash
# QuantTerm — Mac ko 24/7 server banao (ek command, ₹0, koi card nahi).
#
#   cd ~/0to100 && bash deploy/setup_mac.sh
#
# Kya karta hai:
#   1. Mac ki neend band (system sleep off; display so sakta hai)
#   2. launchd service — login/boot pe khud start, crash pe khud restart
#      (Linux ke systemd ka Mac-wala bhai)
# Idempotent — dobara chalao toh sirf reload karega.
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.quantterm.app.plist"
LABEL="com.quantterm.app"

echo "== [1/3] Sleep band (sudo maangega) =="
sudo pmset -a sleep 0 displaysleep 10
echo "   system sleep OFF · display 10 min mein soyega (theek hai)."
echo "   ⚠️  MacBook ho toh: charger lagao aur DHAKKAN KHULA rakho,"
echo "       ya closed-lid ke liye:  sudo pmset -a disablesleep 1"

echo "== [2/3] launchd service =="
mkdir -p "$APP_DIR/logs"
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-lc</string>
        <string>cd "$APP_DIR" &amp;&amp; exec python3 -m streamlit run app.py --server.port 8501 --server.headless true</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>10</integer>
    <key>StandardOutPath</key><string>$APP_DIR/logs/streamlit.log</string>
    <key>StandardErrorPath</key><string>$APP_DIR/logs/streamlit.log</string>
</dict>
</plist>
PL

echo "== [3/3] Service load =="
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load -w "$PLIST"
sleep 3
if curl -sf http://localhost:8501 >/dev/null 2>&1; then
    echo "   ✅ zinda — http://localhost:8501"
else
    echo "   ⏳ start ho raha hai… logs: tail -f $APP_DIR/logs/streamlit.log"
fi

cat <<'DONE'

✅ QuantTerm ab Mac pe 24/7 hai:
   - login/boot pe khud start
   - crash ho toh 10s mein khud restart
   - Mac ab soyega nahi (display bhale so jaye)

Phone se access (ghar ke WiFi ke bahar bhi) — Tailscale (free):
   Mac + phone dono pe Tailscale app → phir http://<mac-tailscale-ip>:8501

Roz subah (sirf NSE live ke liye):  cd ~/0to100 && python3 main.py login
Logs:     tail -f ~/0to100/logs/streamlit.log
Band:     launchctl unload ~/Library/LaunchAgents/com.quantterm.app.plist
Update:   cd ~/0to100 && git pull && launchctl kickstart -k gui/$(id -u)/com.quantterm.app
DONE
