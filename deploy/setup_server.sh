#!/usr/bin/env bash
# Install/update QuantTerm as two supervised processes: complete desk stack + paper-only autonomy.
# Deploys the current checkout. Do not pin historical research branches.
set -euo pipefail

REPO_URL="${QT_REPO_URL:-https://github.com/rishabhdoshi09/0to100.git}"
SCRIPT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -d "$SCRIPT_ROOT/.git" ] || [ -f "$SCRIPT_ROOT/scripts/run_quantterm_complete.sh" ]; then
  APP_DIR="${QT_DIR:-$SCRIPT_ROOT}"
else
  APP_DIR="${QT_DIR:-$HOME/0to100}"
fi
RUN_USER="${QT_USER:-$(id -un)}"
PYTHON_BIN="$APP_DIR/venv/bin/python"

if [ -d "$APP_DIR/.git" ]; then
  if [ -n "${QT_BRANCH:-}" ]; then
    git -C "$APP_DIR" fetch origin "$QT_BRANCH"
    git -C "$APP_DIR" checkout "$QT_BRANCH"
    git -C "$APP_DIR" pull --ff-only origin "$QT_BRANCH"
  fi
elif [ -n "${QT_BRANCH:-}" ]; then
  git clone --branch "$QT_BRANCH" "$REPO_URL" "$APP_DIR"
else
  git clone "$REPO_URL" "$APP_DIR"
fi

BRANCH="$(git -C "$APP_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo current)"
echo "== QuantTerm two-process install: checkout $BRANCH at $APP_DIR =="
sudo timedatectl set-timezone Asia/Kolkata 2>/dev/null || true
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip build-essential cmake curl

cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
"$PYTHON_BIN" -m pip install --upgrade pip wheel
"$PYTHON_BIN" -m pip install -r requirements.txt
[ -f .env ] || { cp .env.example .env 2>/dev/null || touch .env; }
chmod 600 .env 2>/dev/null || true
mkdir -p logs/autonomy logs/intelligence logs/snapshots logs/kite_history logs/product

sudo tee /etc/systemd/system/quantterm-ui.service >/dev/null <<UNIT
[Unit]
Description=QuantTerm complete desk (Vite :5173 + API :8765 + reports :8766)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment=TZ=Asia/Kolkata
Environment=PYTHONUNBUFFERED=1
Environment=PYTHONPATH=$APP_DIR
Environment=QT_NONINTERACTIVE=1
EnvironmentFile=-$APP_DIR/.env
ExecStart=/bin/bash $APP_DIR/scripts/run_quantterm_complete.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

sudo tee /etc/systemd/system/quantterm-autonomy.service >/dev/null <<UNIT
[Unit]
Description=QuantTerm autonomous paper supervisor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment=TZ=Asia/Kolkata
Environment=PYTHONUNBUFFERED=1
Environment=QT_AUTONOMY_OWNER=1
EnvironmentFile=-$APP_DIR/.env
ExecStart=$PYTHON_BIN main.py autonomy --interval 15
Restart=on-failure
RestartSec=5
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
UNIT

# Migrate away from Streamlit and the obsolete combined service.
sudo systemctl disable --now quantterm.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/quantterm.service
sudo systemctl daemon-reload
sudo systemctl enable --now quantterm-autonomy.service quantterm-ui.service
sleep 3
sudo systemctl --no-pager --lines=8 status quantterm-autonomy.service || true
sudo systemctl --no-pager --lines=8 status quantterm-ui.service || true

cat <<DONE

QuantTerm installed as two services on checkout $BRANCH.
The UI service owns the complete stack (desk :5173, API :8765, reports :8766, market-ops).
Daily login: cd "$APP_DIR" && "$PYTHON_BIN" main.py login
Desk logs:     journalctl -u quantterm-ui -f
Autonomy logs: journalctl -u quantterm-autonomy -f
Desk:          http://<server-ip>:5173
DONE
