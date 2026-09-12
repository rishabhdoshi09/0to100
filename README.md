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

## Run on the production Mac

The product UI is the **Vite/React desk**. Streamlit is not the product path.
For the production Mac, use the production branch and the host installer so the
service is pinned to the exact checkout it validates:

```bash
cd ~/0to100
git fetch origin
git checkout claude/build-ai-trading-system-miHHd
git pull --ff-only origin claude/build-ai-trading-system-miHHd
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

Then open `http://127.0.0.1:5173`.

After a later `git pull`, run `bash scripts/install_quantterm_host.sh` again so
the installed service is deliberately repinned and revalidated. Full operating,
upgrade, sleep/lid, and rollback guidance is in
[`docs/MAC_LOCAL_RUNBOOK.md`](docs/MAC_LOCAL_RUNBOOK.md).

### Foreground mode

For a quick session without installing the always-on service, one command owns
the local stack (desk, terminal API, report API, autonomy, and market-operations
worker):

```bash
bash scripts/run_quantterm_complete.sh
```

That is the normal foreground command. Home at `http://127.0.0.1:5173` is the
rest of the day. On an interactive local machine the launcher opens Home once
after the desk is reachable. `./quantterm.sh` is a tiny exec wrapper of the same
command.

Paste the full Kite redirect URL when asked — you do not need to pick out
`request_token` by hand. Broker credentials are optional for non-broker PAPER /
SHADOW research lanes; missing capabilities must degrade honestly rather than
fabricate market data.

`scripts/run_desk.sh` is only a compatibility wrapper. It execs
`scripts/run_quantterm_complete.sh`. Do not start Streamlit, and do not start a
second terminal or checkout for the same stack.

For a new clone:

```bash
git clone https://github.com/rishabhdoshi09/0to100.git && cd 0to100
git checkout claude/build-ai-trading-system-miHHd
cp .env.example .env          # optional: put KITE_API_KEY and KITE_API_SECRET here
chmod 600 .env
bash scripts/run_quantterm_complete.sh
```

## Host controls

| Action | Command |
|---|---|
| Install / repin / validate | `bash scripts/install_quantterm_host.sh` |
| Status | `bash scripts/quantterm_status.sh` |
| Restart | `bash scripts/quantterm_restart.sh` |
| Stop | `bash scripts/quantterm_stop.sh` |
| Foreground complete stack | `bash scripts/run_quantterm_complete.sh` |

The always-on host deployment uses persistent runtime state and platform service
management (`launchd` on macOS, systemd user service on Linux). The supported Mac
procedure is the runbook above; older deployment helpers are not the canonical
production path.

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

*Build: see `VERSION`. Canonical foreground launcher: `bash scripts/run_quantterm_complete.sh`.
Canonical always-on Mac path: `bash scripts/install_quantterm_host.sh`.*
