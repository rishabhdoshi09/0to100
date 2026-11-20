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

## Your scope (Run C): statistical alpha budget, evidence provenance, BUY admission, risk-control bypass

Use finding IDs C-1, C-2, …

1. **The global alpha budget and repeated testing.**
   - Files: `research/alpha_ledger.py` (`reserve`, `verify`, migration, the verify cache, `QT_ALPHA_LEDGER`), `scan/signal_backtest.py` (when a look is reserved; Holm at `REPORT_ALPHA`; which signals and combos form the family; save vs discard; commit `ae55b791`), and `research/overlap_inference.py` (the dependence-aware lower bound and the block count).
   - Find any path that:
     - tests a hypothesis without spending a look;
     - spends a look but later lets a different family or report use it;
     - re-runs selection after seeing results;
     - lets the family size shrink (fewer hypotheses → a weaker correction);
     - lets an env override, test isolation or a missing ledger relax the gate in production;
     - lets the year boundary or a clock change refund budget.
   - Check that the stated guarantee (P(any false BUY admission per IST year) ≤ 5% across program versions) actually follows from the code, and name every lane that is outside it.

2. **Evidence provenance and BUY admission.**
   - Files: `scan/unified_scanner.py` (`_evidence_verdict`, `evidence_provenance_note`, every place a verdict becomes "BUY"), `scan/calibration_snapshot.py`, `product/scan_store.py` (`load_scan`, `bind_evidence`), `product/buy_admission.py`, `product/home_os.py`, `product/recommendations_workspace.py`, `product/decision_ranking.py`, `product/conviction.py`, `scan/conviction.py`, `scan/breakout_sniper.py`, `research/autonomy/telegram_notifications.py`.
   - Enumerate EVERY code path that can label, display, notify or trade a row as BUY (grep for `"BUY"`, `'BUY'`, `verdict`, `UNIFIED_BUY`, `ELIGIBLE_BUY`, `qualified`). For each, prove it passes through the gate, or give a counterexample where an unproven setup is shown or treated as a qualified BUY. Include:
     - the F&O paper path (claimed to be labelled research);
     - the paper EXPLORATION lane;
     - the sniper;
     - replay/historical scans (`evidence_gate=False`);
     - Telegram;
     - the React desk API (`terminal_product_api.py` and the routers it includes).

3. **Risk-control bypass.**
   - Files: `risk/position_sizer.py`, `risk/portfolio_risk.py`, `product/paper_autopilot.py`, `research/auto_research/paper_book.py`, `execution/trade_executor.py`, `execution/autopilot.py`, `alerts/telegram_actions.py`, `product/live_execution_interlock.py`, `product/live_safety.py`, `product/live_readiness.py`, `product/paper_status.py` (cross-engine duplicate and open-risk checks).
   - Look for:
     - any path that can place a live broker order, or that turns a Telegram tap into a live order;
     - any paper path that skips the 1% / 10% / 5% / max-positions limits, or the duplicate check across the two paper engines (legacy `logs/trades.db` vs `intel_book.json`);
     - sizing on a stale price;
     - an exploration lane exceeding its 1/day and 0.5% limits;
     - restart races that open the same symbol twice.
