#!/usr/bin/env bash
# Install/update QuantTerm as two supervised processes: read-only UI + paper-only autonomy.
set -euo pipefail

REPO_URL="${QT_REPO_URL:-https://github.com/rishabhdoshi09/0to100.git}"
BRANCH="${QT_BRANCH:-overhaul/evidence-lab}"
APP_DIR="${QT_DIR:-$HOME/0to100}"
RUN_USER="${QT_USER:-$(id -un)}"
PYTHON_BIN="$APP_DIR/venv/bin/python"
STREAMLIT_BIN="$APP_DIR/venv/bin/streamlit"

echo "== QuantTerm two-process install: $BRANCH =="
sudo timedatectl set-timezone Asia/Kolkata 2>/dev/null || true
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip build-essential cmake curl

if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch origin "$BRANCH"
  git -C "$APP_DIR" checkout "$BRANCH"
  git -C "$APP_DIR" pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
"$PYTHON_BIN" -m pip install --upgrade pip wheel
"$PYTHON_BIN" -m pip install -r requirements.txt
[ -f .env ] || { cp .env.example .env 2>/dev/null || touch .env; }
chmod 600 .env 2>/dev/null || true
mkdir -p logs/autonomy logs/intelligence logs/snapshots logs/kite_history logs/product

sudo tee /etc/systemd/system/quantterm-ui.service >/dev/null <<UNIT
[Unit]
Description=QuantTerm retail UI (read-only control room)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$RUN_USER
WorkingDirectory=$APP_DIR
Environment=TZ=Asia/Kolkata
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=-$APP_DIR/.env
ExecStart=$STREAMLIT_BIN run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
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

# Migrate away from the obsolete combined Streamlit+daemon service.
sudo systemctl disable --now quantterm.service 2>/dev/null || true
sudo rm -f /etc/systemd/system/quantterm.service
sudo systemctl daemon-reload
sudo systemctl enable --now quantterm-autonomy.service quantterm-ui.service
sleep 3
sudo systemctl --no-pager --lines=8 status quantterm-autonomy.service || true
sudo systemctl --no-pager --lines=8 status quantterm-ui.service || true

cat <<DONE

QuantTerm installed as two services on branch $BRANCH.
Daily login: cd "$APP_DIR" && "$PYTHON_BIN" main.py login
UI logs:       journalctl -u quantterm-ui -f
Autonomy logs: journalctl -u quantterm-autonomy -f
UI:            http://<server-ip>:8501
DONE
