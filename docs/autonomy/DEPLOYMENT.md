# QuantTerm — two-process deployment

The retail UI and the autonomy supervisor run as **separate supervised processes**. The supervisor
operates without a browser; the UI is a read-only control room over the supervisor's status snapshot.
One service failing does not corrupt the other, and each restarts independently.

Branch of record: **`overhaul/evidence-lab`**.

## Processes
| Process | Command | Role |
|---------|---------|------|
| Autonomy supervisor | `python main.py autonomy --interval 15` | durable job loop, paper-only, live locked |
| Desk UI | `bash scripts/run_quantterm.sh` | Vite :5173 + API :8765 (never starts Streamlit) |

The daily human action is the normal Zerodha login (`python main.py login`) — Zerodha's auth model
requires it. After the token is persisted, the supervisor picks it up on its next auth-health job (or
a supervised restart); it never logs the token.

## Linux (systemd)
```bash
sudo cp deploy/quantterm-autonomy.service deploy/quantterm-ui.service /etc/systemd/system/
sudo mkdir -p /var/log/quantterm
sudo systemctl daemon-reload
sudo systemctl enable --now quantterm-autonomy quantterm-ui
sudo systemctl status quantterm-autonomy      # SIGINT stop is graceful (preserves job/state)
```

## macOS (launchd)
```bash
cp deploy/com.quantterm.autonomy.plist deploy/com.quantterm.ui.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.quantterm.autonomy.plist
launchctl load ~/Library/LaunchAgents/com.quantterm.ui.plist
```

## Health & safety
- The supervisor holds a single-instance lock (`logs/autonomy/supervisor.lock`); a second instance
  refuses to start.
- Durable state: `logs/autonomy/jobs.db` (job ledger), `state.json` (state machine), `status.json`
  (UI-facing snapshot), `dialogue.jsonl` (typed records). A restart resumes without duplicating work
  and without losing open paper positions.
- Live execution is **disabled** in this milestone; no service can place a broker order.
- Logs are separate per service; secrets are never printed.

## Owner controls
Retail buttons write durable requests to `logs/autonomy/controls.db`. The autonomy service processes
those requests under its single-instance lock. Opening additional browser sessions cannot create
additional schedulers or mutation owners.
