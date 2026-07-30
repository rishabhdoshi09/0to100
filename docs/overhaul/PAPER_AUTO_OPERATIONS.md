# PAPER_AUTO — Operator Runbook

How to run QuantTerm's autonomous paper-trading loop on genuine NSE data. Documents only the
existing system. Routine trades are automatic — you never approve individual paper trades.

## 1. Required genuine NSE files
- **Daily equity bhavcopy** (one file per session), any of the supported forms: `.csv`,
  `.csv.gz`, `.csv.zip`, a `.zip` of CSVs, or a **zip-of-zips** (e.g. NSE's
  `BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip` inside a daily package). Nested archives are
  unpacked automatically.
- **Columns** (modern or legacy names both accepted via aliasing): symbol (`TckrSymb`/`SYMBOL`),
  series (`SctySrs`/`SERIES`), trade date (`TradDt`/a `DDMMYYYY` filename), OHLC
  (`OpnPric/HghPric/LwPric/ClsPric` or `OPEN_PRICE/…`), volume (`TtlTradgVol`/`TTL_TRD_QNTY`).
- **Date coverage**: at least ~130 sessions per name for the momentum family to earn in-sample
  evidence; more is better. Single-symbol families (breakout/trend/pullback) need ~50.
- **Benchmark** (optional but required for `relative_strength`/`sector_rotation`): a Nifty index
  daily series. Without it those families report `MISSING_BENCHMARK` and stay inactive.

The environment ships **no** market data. Nothing is fabricated or downloaded.

## 2. Import procedure
Use Historical Data Setup (UI) or, headless, the existing bridge:
```python
from research.momentum_breakout import data_setup as D          # normalize/extract
from research.intelligence.data.from_bhav import snapshot_from_bhav_dir
from research.intelligence.data.snapshot_store import SnapshotStore

D.safe_extract_zip(open("nse_package.zip","rb"), "staged/")       # nested .csv.zip handled
store = SnapshotStore()                                            # logs/snapshots/
sid, report = snapshot_from_bhav_dir("staged/bhav", store, index_dir="staged/index")
print(report)   # accepted / quarantined / duplicates / future_dated
```
Defective rows are quarantined; duplicate `(symbol, date)` rejected; future-dated sessions
skipped. An empty/all-defective import commits **nothing**.

## 3. Snapshot activation
```python
store.activate_snapshot(sid, actor="user", reason="daily import")   # atomic; audited
```
Activation verifies the manifest checksum + data-file hash first. The active pointer is
`logs/snapshots/ACTIVE`; a crash leaves the old OR new snapshot active, never a partial state.

## 4. Configure paper capital & limits
Paper capital + friction live in the paper book (`PaperBook`: 1% risk/trade, 10% per name, 5%
total open risk, India cash-equity costs + slippage + gap-through-stop). The persisted
`logs/intelligence/paper_config.json` carries the enable flag and starting capital. **This is a
paper configuration, not real-money authorization.**

## 5. Enable PAPER_AUTO (once)
```python
from research.auto_research.scheduler import get_brain
get_brain().enable_paper_auto()     # persisted; survives restart, no re-click
```
Default is enabled. The background worker then runs eligible cycles automatically.

## 6. Verify the worker is running
`get_brain().state.running` is `True`; `state.cycles_run` / `state.last_intel_cycle` advance.
In the app, the **Brain Observatory** shows mode, last cycle, and recent canonical events.

## 7. Inspect the latest cycle
`get_brain().state.last_intel_cycle` (dict): `positions_opened/closed`, `allocation_decisions`,
`eligibility` (`TRADED` / `NO_ELIGIBLE_TRADE` / `NO_DATA`), `no_action_reasons`. Every canonical
event carries the pinned snapshot id.

## 8. Inspect open paper positions
`get_brain().intel_book.open` (per-strategy positions with stop/target) and
`get_brain().intel_book.stats()` (n_trades, expectancy_R, net_pnl, drawdown).

## 9. Stop new paper entries (keep managing exits)
Regime stand-down does this automatically in RISK_OFF tape. Manually, deactivate the snapshot
(`store` — no active snapshot ⇒ no new entries) or disable PAPER_AUTO (below). Existing positions
continue to be managed/exited while price data is sufficient.

## 10. Disable PAPER_AUTO
```python
get_brain().disable_paper_auto()    # persisted; honoured across restart
```

## 11. Emergency: close paper positions manually
```python
b = get_brain()
b.intel_book.mark({sym: (o, h, l, c) for ...}, today)   # mark to force stop/target/close, or
# clear the persisted book: delete logs/intelligence/intel_book.json while the worker is stopped
```

## 12. Restart behavior
On restart the brain: loads the paper config (enable flag), restores the paper book (open
positions + stops/targets) from `intel_book.json`, reconciles against runtime state, resumes
**exits/management first**, and only opens new entries once data + risk health pass. Completed
cycles are remembered (idempotent) so nothing is duplicated. **No re-approval is required.**

## 13. Safe response to stale data
Freshness is computed from the NSE trading calendar (weekends, holidays, publication cutoff),
not calendar-date math. If the required completed session is missing beyond the publication
allowance, the snapshot is **not FORWARD_ELIGIBLE** → new entries are blocked while existing
positions keep being managed. Future-dated bars block entirely.

## 14. Safe response to corrupt persistence
Corrupt `intel_book.json` / `runtime_state.json` / `paper_config.json` load as empty (never
crash); an unverifiable snapshot fails verification → not opened → no new risk. A pointer to a
missing snapshot resolves to none. In all cases the loop degrades to a safe no-op.

## 15. Paper vs simulator vs live evidence (never blurred)
- **Paper evidence** — outcomes from the PAPER_AUTO loop on a real snapshot; `OutcomeObservation.split == "forward"`. Real market evidence, paper-realistic frictions.
- **In-sample evidence** — the bootstrap backtest of a strategy's own rules on snapshot history
  (`in_sample_trades` on the card); establishes execution/eligibility, **never** forward-confirmed.
- **Simulator evidence** — the EMS `SimBroker` (a different subsystem); **testing infrastructure
  only**, never market evidence.
- **Live evidence** — does not exist; there is no real broker path. `USER_APPROVED` is user-only.

QuantTerm may automatically decide **not** to trade (`NO_ELIGIBLE_TRADE` is a healthy result).
When a qualified paper trade exists it enters, manages, exits and learns without user intervention.
