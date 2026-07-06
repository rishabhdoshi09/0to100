# QuantTerm ko 24/7 Chalana (Always-On Setup)

Mac sleep hote hi background scans, Telegram alerts, breakout sniper —
sab ruk jaata hai. Full-time trading ke liye system ko hamesha jaagna
chahiye. Do raste:

---

## Option A: Sasta VPS (recommended, ~₹300-500/month)

Koi bhi 2GB-RAM Ubuntu VPS chalega (Hetzner/DigitalOcean/Oracle
free-tier bhi).

### 1. Server taiyaar karo
```bash
ssh root@YOUR_SERVER_IP
apt update && apt install -y python3.11 python3.11-venv git
```

### 2. Project daalo
```bash
git clone https://github.com/rishabhdoshi09/0to100.git
cd 0to100
git checkout claude/deepseek-multi-agent-system-nrO7n
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env    # keys bharo
```

### 3. systemd service (crash ho toh khud restart)
`/etc/systemd/system/quantterm.service`:
```ini
[Unit]
Description=QuantTerm Trading Terminal
After=network.target

[Service]
WorkingDirectory=/root/0to100
ExecStart=/root/0to100/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
```bash
systemctl daemon-reload
systemctl enable --now quantterm
systemctl status quantterm        # green = zinda
```

### 4. Phone/laptop se kholo
`http://YOUR_SERVER_IP:8501` — ya Tailscale laga lo (free) taaki
sirf tumhare devices se khule:
```bash
curl -fsSL https://tailscale.com/install.sh | sh && tailscale up
```

---

## Option B: Ghar ka Raspberry Pi / purana laptop

Wahi steps — bas machine ghar pe. Bijli + internet stable ho toh
kaafi hai. Mac pe hi chalana ho toh kam se kam sleep band karo:
```bash
sudo pmset -a sleep 0 displaysleep 10
```

---

## Roz ka ritual (server pe bhi wahi)

Kite token roz subah chahiye. Server pe:
```bash
ssh root@YOUR_SERVER_IP
cd 0to100 && source venv/bin/activate && python main.py login
```
Token daalte hi service khud naya token utha legi (`.env` reload
next cycle pe). 8:30 baje Telegram reminder waise bhi aayega agar
bhool gaye.

## Updates lena
```bash
cd 0to100 && git pull origin claude/deepseek-multi-agent-system-nrO7n
systemctl restart quantterm
```

## Health check
- Telegram pe subah Pulse aa raha hai? → scans chal rahe hain
- App ke scanner header pe sab dots green? → data sources theek
- `journalctl -u quantterm -f` → live logs
