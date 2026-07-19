#!/usr/bin/env bash
# QuantTerm — Oracle/Ubuntu server one-shot setup.
#
#   curl -fsSL <raw-url>/deploy/setup_server.sh | bash
#   ya repo clone karke:  bash deploy/setup_server.sh
#
# Kya karta hai: deps → swap → repo → venv → pip install → systemd service.
# Idempotent — dobara chalane pe update + restart hi karega, todega nahi.
set -euo pipefail

REPO_URL="${QT_REPO_URL:-https://github.com/rishabhdoshi09/0to100.git}"
BRANCH="${QT_BRANCH:-claude/deepseek-multi-agent-system-nrO7n}"
APP_DIR="${QT_DIR:-$HOME/0to100}"
RUN_USER="$(whoami)"

echo "== [0/6] Timezone → IST (market ka time hi sach hai) =="
sudo timedatectl set-timezone Asia/Kolkata 2>/dev/null \
    && echo "   Asia/Kolkata set." \
    || echo "   ⚠️  timezone set nahi hua (container?) — koi baat nahi, gates IST-explicit hain."

echo "== [1/6] System packages =="
sudo apt-get update -y
sudo apt-get install -y git python3 python3-venv python3-pip \
    build-essential cmake curl

echo "== [2/6] Swap (2G) — chhote instances OOM na hon =="
# OpenVZ/LXC container-VPS swap banane nahi dete — wahan fail hona OK hai
# (4GB RAM ho toh swap optional hai); script rukna nahi chahiye.
if ! swapon --show | grep -q .; then
    if sudo fallocate -l 2G /swapfile 2>/dev/null \
        && sudo chmod 600 /swapfile \
        && sudo mkswap /swapfile >/dev/null \
        && sudo swapon /swapfile 2>/dev/null; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
        echo "   swap on."
    else
        sudo rm -f /swapfile
        echo "   ⚠️  swap nahi ban paya (container VPS?) — 4GB RAM pe theek hai, aage badh rahe."
    fi
else
    echo "   swap already on — skip."
fi

echo "== [3/6] Repo =="
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" fetch origin "$BRANCH"
    git -C "$APP_DIR" checkout "$BRANCH"
    git -C "$APP_DIR" pull origin "$BRANCH"
else
    # Private repo? Pehle: export QT_REPO_URL=https://<TOKEN>@github.com/rishabhdoshi09/0to100.git
    git clone --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi

echo "== [4/6] Python venv + dependencies =="
cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip wheel
./venv/bin/pip install -r requirements.txt

echo "== [5/6] .env =="
if [ ! -f .env ]; then
    cp .env.example .env 2>/dev/null || touch .env
    echo "   ⚠️  .env banaya — API keys bharna mat bhoolna:  nano $APP_DIR/.env"
else
    echo "   .env already hai — untouched."
fi

echo "== [6/6] systemd service =="
sudo tee /etc/systemd/system/quantterm.service >/dev/null <<UNIT
[Unit]
Description=QuantTerm Trading Terminal
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$APP_DIR
ExecStart=$APP_DIR/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10
Environment=TZ=Asia/Kolkata
OOMScoreAdjust=-100

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now quantterm
sleep 3
sudo systemctl --no-pager --lines=5 status quantterm || true

cat <<'DONE'

✅ QuantTerm service zinda hai.

Agla (ek baar):
  1. nano ~/0to100/.env         # Kite/Telegram/DeepSeek keys
  2. sudo systemctl restart quantterm
  3. Access (recommended — Tailscale, port kholne ki zaroorat nahi):
       curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
     phir apne devices se:  http://<tailscale-ip>:8501

Roz subah (sirf NSE live ke liye):
  cd ~/0to100 && ./venv/bin/python main.py login

Logs:      journalctl -u quantterm -f
Update:    cd ~/0to100 && git pull && sudo systemctl restart quantterm
DONE
