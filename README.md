# ⚡ QuantTerm

**An evidence-driven, self-learning trading terminal for NSE (live via Zerodha) and US equities (paper)** — whole-market scanning, expected-value ranking, a gated autopilot, and a Brain that reads every subsystem and delivers one verdict a day to your phone.

> The edge is not a better breakout pattern. The edge is a system that **measures, allocates, and retires** its own strategies faster and more honestly than its operator's emotions would.

---

## What it does (one loop)

```text
data → signals → EV ranking → risk gates → execution → outcomes → learning
  ▲                                                                  │
  └──────────────── every closed trade recalibrates ─────────────────┘
```

- **Scan** the full NSE universe every 15 min (and US indices on demand): 16 signals — confirmed breakouts (ATR + volume), VCP, patterns, pre-breakout, pullback, momentum.
- **Rank by Expected Value**, not points: `EV% = [P(win)×avgWin − P(loss)×avgLoss] × setup risk`, from the system's **own forward-tested outcomes**, Wilson-shrunk so big samples outrank lucky streaks.
- **Gate** every trade: score, edge, sector, regime, live-price anchor, daily/position/sector caps, circuit breaker, and survival vetoes.
- **Learn**: every BUY and every rejected candidate is journaled with its prediction; outcomes resolve later; calibration is audited; leaky signals get demoted per regime.
- **Stay honest**: no market evidence means no profitability claim. Stale, missing, degraded, or blocked data must look that way in both the APIs and the desk.

## Canonical local product path

The product UI is the **Vite/React desk**. Streamlit is not the product path.

For an interactive foreground session:

```bash
cd ~/0to100
bash scripts/run_quantterm_complete.sh
```

The desk is at `http://127.0.0.1:5173`. `./quantterm.sh` and `scripts/run_desk.sh` are compatibility wrappers around the same complete-stack launcher. Do not start a second copy of the stack in another terminal.

First-time clone/configuration:

```bash
git clone https://github.com/rishabhdoshi09/0to100.git
cd 0to100
git checkout claude/build-ai-trading-system-miHHd
cp .env.example .env
chmod 600 .env
# add KITE_API_KEY and KITE_API_SECRET to .env
bash scripts/run_quantterm_complete.sh
```

## Canonical MacBook always-on path

For the production MacBook, use the dedicated host installer introduced by the persistent host deployment:

```bash
cd ~/0to100
git fetch origin
git checkout claude/build-ai-trading-system-miHHd
git pull --ff-only origin claude/build-ai-trading-system-miHHd
bash scripts/install_quantterm_host.sh
bash scripts/quantterm_status.sh
```

The service is pinned to the exact validated checkout SHA and stores durable runtime state outside the Git checkout under `~/Library/Application Support/QuantTerm/runtime`.

After pulling a new production SHA, rerun `bash scripts/install_quantterm_host.sh` so the installed service is validated and pinned to that exact revision.

**Rollback is not atomic.** Deployment runs from a mutable Git checkout. To revert, explicitly check out the previous known-good SHA and reinstall the service; do not force a SHA-mismatched service to start.

Full Mac operating procedure: [`docs/MACBOOK_RUNBOOK.md`](docs/MACBOOK_RUNBOOK.md).

## Invariants (the non-negotiables)

1. **No fake data, ever** — no data → skip, never simulate market truth.
2. **Stale must look stale** — freshness is labeled everywhere.
3. **Every live entry must ship with its exchange-side protective exit**; deployment changes must not bypass the execution safety boundary.
4. **PAPER/SHADOW is the proving ground**; live execution stays locked until separately authorised and verified.
5. **Telegram actions remain paper-only.**
6. **1% risk/trade · 10% per name · 5% total open risk.**
7. **Evidence over vibes** — fewer than 30 outcomes means no performance claim; proven losers demote.
8. **Every decision is outcome-tracked** — including rejections.

## Testing

```bash
python -m pytest tests/
```

CI runs the canonical hermetic suite, launch/supervisor validation, compile-all, controlled integration checks, and a non-blocking live-source lane. Terminal UI and Product Acceptance are separate release gates.

Issue #92 live Definition of Done (stack must already be running):

```bash
python scripts/verify_issue92_dod.py
```

That writes `docs/issue92_live_dod_proof.md` and `docs/issue92_live_dod_proof.json` against the real local API, including the tested git SHA. It does not mock handlers.

## Architecture

The full map lives in [`CLAUDE.md`](CLAUDE.md) — data layer (bhavcopy/Kite/NSE/US), signal layer (scanner, EV engine, live edge, breadth), risk layer (sizer, portfolio risk, correlation), execution boundary, core (Brain, decision journal, sim lab, market clock), and the **Vite/React desk** (Home · Market Scanner · Recommendations · Market Reports · Stock Intelligence).

Historical research branches such as `overhaul/evidence-lab` are **not the current product path**.

*Build: see `VERSION`. Canonical foreground launcher: `bash scripts/run_quantterm_complete.sh`. Canonical persistent Mac install: `bash scripts/install_quantterm_host.sh`.*
