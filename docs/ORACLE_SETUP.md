# Oracle Cloud Free Tier pe QuantTerm 24/7 (₹0/month)

> ⚠️ **Oracle signup ke liye credit card zaroori hai** (verification —
> charge nahi hota, par card ke bina account banta hi nahi). Card nahi
> hai? **[ALWAYS_ON.md](ALWAYS_ON.md) → Option B**: apna Mac ek command
> mein 24/7 (`deploy/setup_mac.sh`, ₹0) — ya UPI-waala VPS (Hostinger).

Oracle ka **Always Free** tier hamesha free hai (trial nahi):
**Ampere A1 (ARM): 4 OCPU + 24GB RAM tak** — QuantTerm ke liye kaafi se zyada.
Mumbai region NSE ke liye best latency deta hai.

---

## Step 1 — Account (10 min, browser)

1. <https://oracle.com/cloud/free> → Sign up.
2. **Home region: India West (Mumbai)** chuno — baad mein badal NAHI sakte.
3. Card verification hota hai (charge nahi) — Always Free pe kabhi bill
   nahi banta jab tak tum khud paid upgrade na karo.

## Step 2 — VM banao (5 min)

1. Console → **Compute → Instances → Create Instance**.
2. Image: **Ubuntu 24.04** (aarch64).
3. Shape: **Ampere → VM.Standard.A1.Flex → 4 OCPU / 24 GB** ("Always Free
   eligible" tag dikhna chahiye).
4. SSH key: apni public key daalo (ya download karo jo woh banaye).
5. Create. Public IP note kar lo.

> **"Out of capacity" aaye toh:** Mumbai mein A1 kabhi-kabhi full hota
> hai. (a) 2 OCPU / 12GB try karo, (b) doosri Availability Domain,
> (c) thodi der baad retry — 1-2 din mein mil hi jaata hai. Impatient ho
> toh VM.Standard.E2.1.Micro (x86, 1GB) se shuru karo — swap script
> laga deta hai, chal jayega (dheema).

## Step 3 — Server setup (ek command)

```bash
ssh ubuntu@YOUR_PUBLIC_IP

# private repo hai, isliye GitHub token ke saath (Settings → Developer
# settings → Personal access tokens → repo read):
export QT_REPO_URL=https://YOUR_TOKEN@github.com/rishabhdoshi09/0to100.git
curl -fsSL https://raw.githubusercontent.com/rishabhdoshi09/0to100/claude/deepseek-multi-agent-system-nrO7n/deploy/setup_server.sh -o setup.sh
bash setup.sh
```

(Ya repo pehle clone karke `bash deploy/setup_server.sh`.)

Script khud karta hai: packages → 2G swap → clone/pull → venv →
`pip install` → **systemd service** (crash pe 10s mein auto-restart,
reboot pe auto-start).

## Step 4 — Keys + restart

```bash
nano ~/0to100/.env          # KITE_*, TELEGRAM_*, DEEPSEEK_API_KEY
sudo systemctl restart quantterm
```

## Step 5 — Access: Tailscale (recommended)

Oracle ke firewall/security-list mein kuch kholne ki zaroorat nahi —
Tailscale private network bana deta hai (free, 2 min):

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up          # login link kholo
tailscale ip -4            # yeh IP note karo
```

Phone/Mac pe bhi Tailscale app → phir kahin se bhi:
`http://<tailscale-ip>:8501`

<details>
<summary>Public port kholna ho toh (kam secure — avoid)</summary>

1. Console → VCN → Security List → Ingress rule: TCP 8501, source 0.0.0.0/0
2. Server pe Oracle ka baked-in iptables bhi kholo:
   ```bash
   sudo iptables -I INPUT -p tcp --dport 8501 -j ACCEPT
   sudo netfilter-persistent save
   ```
</details>

## Roz ka ritual

- **Kite token (sirf NSE live ke liye):**
  `ssh ubuntu@IP` → `cd 0to100 && ./venv/bin/python main.py login`
  Bhool jao toh 8:30 pe Telegram reminder aata hai. Paper mode + US
  scanning + EOD data token ke **bina bhi** chalte hain.
- **Zinda hai?** Subah Brain briefing + Pulse Telegram pe aaye = daemons
  zinda. App → Diagnostics → System Pulse dots green.

## Maintenance

```bash
# update
cd ~/0to100 && git pull origin claude/deepseek-multi-agent-system-nrO7n
sudo systemctl restart quantterm

# logs
journalctl -u quantterm -f

# service control
sudo systemctl status|restart|stop quantterm
```
