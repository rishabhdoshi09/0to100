# ⚡ QuantTerm

**An evidence-driven, self-learning trading terminal for NSE (live via
Zerodha) and US equities (paper)** — whole-market scanning, expected-value
ranking, a gated autopilot, and a Brain that reads every subsystem and
delivers one verdict a day to your phone.

> The edge is not a better breakout pattern. The edge is a system that
> **measures, allocates, and retires** its own strategies faster and more
> honestly than its operator's emotions would.

---

## What it does (one loop)

```
data → signals → EV ranking → risk gates → execution → outcomes → learning
  ▲                                                                  │
  └──────────────── every closed trade recalibrates ─────────────────┘
```

- **Scan** the full NSE universe every 15 min (and US indices on demand):
  16 signals — confirmed breakouts (ATR + volume), VCP, patterns,
  pre-breakout, pullback, momentum.
- **Rank by Expected Value**, not points: `EV% = [P(win)×avgWin −
  P(loss)×avgLoss] × setup risk`, from the system's **own forward-tested
  outcomes**, Wilson-shrunk so big samples outrank lucky streaks.
- **Gate** every trade: 15 checks (score, edge, sector, regime, live-price
  anchor, daily/position/sector caps, circuit breaker, Brain survival veto).
- **Execute** with an exchange-side GTT OCO exit on every entry. Paper
  mode is the default and the proving ground.
- **Learn**: every BUY and every *rejected* candidate is journaled with its
  prediction; outcomes resolve in 5 sessions; calibration is audited
  ("70% bola toh 70% nikla?"); leaky signals get demoted per-regime.
- **🧠 The Brain** composes regime × live edge × breadth × options
  positioning × portfolio risk × correlation × autopilot health into one
  posture (GREEN LIGHT / NORMAL / DEFENSIVE / STAND ASIDE) + a prioritised
  to-do — on the Pulse tab and in a morning Telegram briefing.

## Canonical product path (one command)

The product UI is the **Vite/React desk**. Streamlit is not the product path
and is not started. One command owns the local stack (desk, terminal API,
report API, autonomy, and the market-operations worker):

```bash
cd ~/0to100
bash scripts/run_quantterm_complete.sh
```

Open `http://127.0.0.1:5173`. Paste the full Kite redirect URL when asked — you do not need to pick out `request_token` by hand.

`scripts/run_desk.sh` is only a compatibility wrapper. It execs
`scripts/run_quantterm_complete.sh`. Do not start Streamlit, and do not start
a second terminal for the same stack.

```bash
git clone https://github.com/rishabhdoshi09/0to100.git && cd 0to100
cp .env.example .env          # put KITE_API_KEY and KITE_API_SECRET in it once
bash scripts/run_quantterm_complete.sh
```

## Run it 24/7

| Where | How |
|---|---|
| Your Mac (₹0) | `bash deploy/setup_mac.sh` — launchd service; add `QT_ECO=1` on a shared/fanless machine |
| Any Ubuntu server / VPS | `bash deploy/setup_server.sh` — systemd, IST timezone, swap, auto-restart |
| Oracle Cloud free tier | see `docs/ORACLE_SETUP.md` |

Details: `docs/ALWAYS_ON.md`.

## Invariants (the non-negotiables)

1. **No fake data, ever** — no data → skip, never simulate.
2. **Stale must look stale** — freshness is labeled everywhere.
3. **Every trade ships with an exchange-side exit** (GTT OCO).
4. **Telegram taps are paper-only**; live orders need the app's ticket.
5. **1% risk/trade · 10% per name · 5% total open risk.**
6. **Evidence over vibes** — <30 outcomes = no claim; proven losers demote.
7. **Every decision is outcome-tracked** — including rejections.

## Testing

```bash
python -m pytest tests/            # money-critical suite, network-free
```

CI runs the suite + `compileall` on every push. New money-path code lands
with tests in `tests/test_money_paths.py` — no exceptions.

Issue #92 live Definition of Done (stack must already be running):

```bash
python scripts/verify_issue92_dod.py
```

That writes `docs/issue92_live_dod_proof.md` and
`docs/issue92_live_dod_proof.json` against the real local API, including
the tested git SHA. It does not mock handlers.

## Architecture

The full map lives in [`CLAUDE.md`](CLAUDE.md) — data layer (bhavcopy/Kite/
NSE/US), signal layer (scanner, EV engine, live edge, breadth), risk layer
(sizer, portfolio risk, correlation), execution (Kite + GTT, autopilot),
core (Brain, decision journal, sim lab, market clock), and the **Vite/React
desk** (Home · Market Scanner · Recommendations · Market Reports · Stock
Intelligence). Archived Streamlit pages under `ui/` are not started.

*Build: see `VERSION`. Canonical launcher: `bash scripts/run_quantterm_complete.sh`.
Historical research branches such as `overhaul/evidence-lab` are not the
current product path.*
