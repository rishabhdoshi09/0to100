# QuantTerm ko 24/7 Chalana (Always-On Setup)

Canonical product: the **Vite/React desk**. One command owns the local stack:
`bash scripts/run_quantterm_complete.sh` → http://127.0.0.1:5173.
Streamlit is not the product path. Historical research branches such as
`overhaul/evidence-lab` are not the current checkout.

Mac sleep hote hi background scans, Telegram alerts, breakout sniper —
sab ruk jaata hai. Full-time trading ke liye system ko hamesha jaagna
chahiye. Do raste:

---

## Option A: Sasta VPS (recommended, ~₹300-500/month)

Koi bhi 2GB-RAM Ubuntu VPS chalega (Hetzner/DigitalOcean/Oracle
free-tier bhi).

> **Oracle Always Free (₹0) chuna hai?** Poora step-by-step —
> account → Ampere A1 VM → ek-command setup script → Tailscale —
> **[docs/ORACLE_SETUP.md](ORACLE_SETUP.md)** mein hai.
> Server-side sab kuch `deploy/setup_server.sh` automate karta hai.

### 1. Server taiyaar karo
```bash
ssh root@YOUR_SERVER_IP
apt update && apt install -y python3.11 python3.11-venv git
```

### 2. Project daalo
```bash
git clone https://github.com/rishabhdoshi09/0to100.git
cd 0to100
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
Environment=PYTHONPATH=/root/0to100
ExecStart=/bin/bash /root/0to100/scripts/run_quantterm.sh
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
`http://YOUR_SERVER_IP:5173` — ya Tailscale laga lo (free) taaki
sirf tumhare devices se khule:
```bash
curl -fsSL https://tailscale.com/install.sh | sh && tailscale up
```

---

## Option B: Apna Mac hi 24/7 (₹0, koi credit card nahi)

> Oracle/AWS/GCP sab **credit card maangte hain** — card nahi hai toh
> yeh sabse practical rasta hai. Ek command:
```bash
cd ~/0to100 && bash deploy/setup_mac.sh
```
Script karta hai: Mac ki sleep band + **launchd service** (login/boot pe
khud start, crash pe 10s mein khud restart — systemd jaisa hi). Phone se
bahar se dekhna ho toh Tailscale (free) laga lo.
- MacBook: charger + dhakkan khula, ya `sudo pmset -a disablesleep 1`
- Kharcha: ~₹100-150/month bijli. Limitation: bijli/net gaya = system gaya.

### 🍃 Mac garam ho raha hai? (business software saath chal raha?)

Eco mode chalao — **wahi signals, wahi gates, bas thandi machine**:
```bash
QT_ECO=1 bash deploy/setup_mac.sh     # service eco mein reinstall
```
Eco kya karta hai: **off-hours full-market scan bilkul band** (raat ko
scan pure heat tha — EOD data badalta hi nahi), scan threads 8→2,
market-hours cadence 30 min. Sniper (instant breakouts) websocket hai —
woh waise hi chalta rehta hai. Briefing/outcomes/backtest sab normal.

MacBook **Air** (fanless) + doosra software 24/7 = long-term sahi nahi.
Sasta permanent fix: **Raspberry Pi 4/5 (₹5-8k one-time, UPI se milta
hai, 5W, silent)** — usi pe `deploy/setup_server.sh` chala do, Mac
business ke liye free.

**Baad mein card ke bina VPS chahiye ho toh:** Hostinger VPS (~₹350/mo)
**UPI accept karta hai** — phir `deploy/setup_server.sh` wahi ek-command
setup wahan chala dena.

## Option C: Ghar ka Raspberry Pi / purana laptop

Wahi Linux steps (`deploy/setup_server.sh`) — bas machine ghar pe.
Bijli + internet stable ho toh kaafi hai.

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
cd 0to100 && git pull
systemctl restart quantterm
```

## Health check
- Telegram pe subah Pulse aa raha hai? → scans chal rahe hain
- App ke scanner header pe sab dots green? → data sources theek
- `journalctl -u quantterm -f` → live logs
