# Independent adversarial code audit: QuantTerm (NSE India paper-trading research terminal)

You are an independent, critical reviewer, not an approver. Another engineer (Claude) wrote most of this code. Your job is to find what is wrong, so do NOT assume claims in commit messages, docstrings, comments or `docs/` are true. Treat every claim as a hypothesis and verify it against the code.

## Target
- Repository: `rishabhdoshi09/0to100`. The working directory is a git checkout at the exact commit `9636b03055961e5f53b5dfb7cfe5cfede03ba13f` (branch `claude/settlement-batch-20261118`).
- Production currently runs `125880a6`. The candidate change set is `125880a6..9636b030` (52 commits). The most recent cycle, not yet reviewed by anyone, is `dd4e9c34..9636b030` (8 commits).
- Review bundle (read-only): `<review-bundle>/commits_since_production.txt`, `<review-bundle>/stat_since_production.txt`, `<review-bundle>/diff_since_last_review.patch`, `<review-bundle>/diff_since_production.patch`.
- You may run read-only commands: `git log/show/diff/blame/grep`, `rg`, `grep`, `sed -n`, `nl`, `cat` on files INSIDE this checkout or BUNDLE. The sandbox is read-only, so pytest probably cannot run (tmp writes fail). If you need a deterministic check, write out the exact pytest test you would run; the coordinator will run it and report back.

## Hard rules
- Do not read `.env` or any file outside this checkout and BUNDLE. In particular, never read `~/.codex`, `~/.claude`, `/Volumes`, `/Users/apple/0to100` (the production checkout) or any runtime/log directory. Never print a token, key or credential.
- Do not attempt any write, network call, broker call or process control.
- Do not soften findings. Equally, do not invent them: every CONFIRMED finding must be backed by a concrete trace through real code.

## Context you need (product invariants the code claims to enforce)
- No fake data. Stale data must look stale. Live broker trading is hard-locked; this system is paper-only. Telegram taps are paper-only.
- One rulebook: `core/evaluation_spec.py`, a 10-official-session horizon with gap-aware fills and exits. Backtest, forward tracker, decision journal, paper book and counterfactuals must all use it.
- A BUY must be admitted only by evidence: a walk-forward with ≥30 trades, a dependence-aware lower bound > 0, Holm family-wise correction at α/52 per weekly report, a GLOBAL 52-look annual alpha budget (`research/alpha_ledger.py`), provenance (report schema, eval spec, horizon, family method, age ≤ 21 days), and re-verification at the scan-store boundary (`product/scan_store.py::load_scan`) against the active calibration snapshot (`scan/calibration_snapshot.py`).
- Risk: 1% risk per trade, 10% per name, 5% total open risk, max 5 positions; a paper EXPLORATION lane (1 trade/day at 0.5% risk) may probe gated setups but must never be labelled as a qualified BUY.
- Learning must count independent units (day, symbol, reason), never copies (`product/learning_observations.py`).

## Required output format (Markdown)
1. `## Findings`: one entry per finding, ordered P0 → P2. Each entry has:
   - **ID** (e.g. A-1), **Severity** P0/P1/P2:
     - P0 = can admit an unproven BUY, bypass a risk control, corrupt evidence or learning, or leak future information.
     - P1 = materially wrong statistics, learning or labelling, without a direct money path.
     - P2 = robustness or clarity.
   - **Status**: CONFIRMED (you traced a concrete counterexample through the code) or HYPOTHESIS (plausible, but not fully traced, or it depends on runtime state).
   - **Location**: `path:line` and the function name(s). Cite real line numbers from this checkout.
   - **Counterexample**: concrete inputs or state → the wrong output.
   - **Deterministic test**: a minimal pytest sketch that would fail today if the finding is real. Name existing fixtures or helpers only when you have verified they exist.
   - **Suggested fix** (brief).
2. `## Verified correct`: claims you checked and found to hold, with file:line evidence. Keep it short.
3. `## Not examined / limits`: what you did not cover.

Be concrete and economical. Quality over quantity: a few real, traced defects beat many vague concerns.

## Your scope (Run B): point-in-time integrity, look-ahead leakage, survivorship and corporate actions

Use finding IDs B-1, B-2, …

Audit whether any decision, backtest or learning record can see information that was not available at its decision time.

1. **Look-ahead in the walk-forward and scanner.**
   - Files: `scan/signal_backtest.py` (`run_backtest` and its window slicing), `scan/unified_scanner.py`, `core/evaluation_spec.py` (`simulate_trade`: entry trigger, gap fills, stop/target ordering within a bar, trail), `research/edge_diagnosis.py`, `research/full_history_probe.py`.
   - Check:
     - whether any indicator, 52-week high, ATR, volume average, RSI, percentile, universe filter or calibration uses bars after the decision session;
     - whether same-bar stop and target ambiguity is resolved optimistically;
     - whether entries fill at prices not traded;
     - whether exits use the closes of sessions after the horizon;
     - whether costs are applied on both legs.

2. **Point-in-time features and the live path.**
   - Files: `product/pit_features.py`, `data/bhavcopy_store.py` (`get_ohlcv`, the intraday overlay, as-of reads, caching), `data/index_store.py`, the regime and breadth inputs used at decision time, and `scan/calibration_snapshot.py`.
   - Check: whether the calibration used for a scan was built from a report generated after that scan's decision date; whether replay or historical scans use today's calibration.

3. **Survivorship.**
   - Files: `data/nse_universe.py::point_in_time_universe` and its history file, and how the walk-forward and the edge diagnosis choose symbols.
   - Check: whether delisted names are present; whether symbols are sampled from the CURRENT store only, which implies survivorship; whether the sampling (e.g. even stride over `store_symbols()`) is survivorship-free; what `survivorship_complete=False` does downstream (fail closed or silent).

4. **Corporate actions.**
   - Files: `data/corporate_actions.py` (`adjust_frame`, `load_events`, coverage), `core/data_integrity.py` (`phantom_gaps`, `verify_ca_adjustment`), and how CA coverage (claimed to be verified only from 2021-08-28) gates which trades count as evidence (STRICT / ENTRY_ONLY / UNCOVERED tiers in `research/edge_diagnosis.py`).
   - Check:
     - whether adjustment is applied on read consistently in backtest and live;
     - whether a missing events file silently counts as adjusted;
     - whether adjusting with events known only later (an ex-date after the decision) leaks future information into indicators;
     - whether volume is adjusted consistently with price.

5. **Calendars.** `data/nse_holidays.json`, any calendar manifest, and `core/market_clock.py`. Check whether the 10-session horizon counts official sessions correctly across holidays, how a holiday-bhavcopy replay is rejected (commit `eda3d95f` on the base), and whether 2027 dates fail closed.
