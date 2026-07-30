# 🧠 Autonomous Research Brain

QuantTerm thinking for itself — a headless loop that studies the market, reasons in the
open, rejects weak ideas, tracks what is decaying or improving, and **proposes** better
strategies, with **zero human intervention** up to one deliberate stop.

## The loop (one cycle)

```
observe data readiness
        ↓
generate readable candidates (approved grammar, seeded, family-diverse)
        ↓
reason through each   →  structural gate (leakage / complexity / point-in-time)
                      →  evidence gate  (sample / concentration / cost / drawdown)
        ↓
reject the weak ones  (every rejection recorded with its reason)
        ↓
shortlist survivors, weigh the multiple-testing burden
        ↓
auto-advance lifecycle  GENERATED → UNDER_REVIEW → PROMISING → AWAITING_USER_APPROVAL
        ↓
STOP.  Park proposals for a human.  Never approve, activate, or trade.
```

Everything the brain thinks is written to an **append-only research thread**
(`OBSERVE → REASON → DECIDE → PROPOSE → CONCLUDE`), so a person can watch exactly *why* each
idea was rejected or shortlisted.

## Full PAPER autonomy (opt-in) — "blow up paper money, but get smarter"

A human can engage **full paper autonomy** (`brain.engage_paper_autonomy()` / the UI toggle
/ `QT_PAPER_AUTONOMY`). Once on, the brain runs the *whole* loop with no human in the loop:

```
survivor (real-data evidence)  →  auto-approve for PAPER (actor: paper_autopilot)
    →  activate  →  PAPER_EVALUATION  →  place SIMULATED trades (PaperBook)
    →  mark against real bars, book realized R  →  learn
    →  autonomously RETIRE proven losers, keep what earns
```

- `research/auto_research/paper_book.py` — a self-contained simulated ledger. It enforces the
  house risk caps (1% risk/trade, 10% per name, 5% total open risk, max positions), marks
  positions stop-first, and books realized R. It imports **no** real-order path — the worst
  it can do is lose imaginary money.
- `research/auto_research/paper_autonomy.py` — deploys survivors, trades them day by day,
  and each cycle retires any strategy whose real paper expectancy has proven negative. That
  daily pruning is how the system "gets smarter by the day".
- It still **refuses synthetic evidence** and a red data gate: full autonomy trades
  real-data strategies on its own; it never fabricates results. With no research-grade data
  it honestly deploys nothing.

## The one boundary that never moves (LIVE)

Full paper autonomy trades **only simulated money and can never reach live.** The
`paper_autopilot` actor may cross the *paper* gates, but the lifecycle **forbids it from the
only transition that leads toward live** — `PAPER_EVALUATION → ELIGIBLE_FOR_LIVE_REVIEW`
stays **user-only**. Moving a strategy toward real money still requires a person, and LIVE
stays migration-locked.

This boundary is **structural**, not a promise:
- `spec._TRANSITIONS`: `ELIGIBLE_FOR_LIVE_REVIEW` is reachable only by `actor="user"`;
  `paper_autopilot` and `system` raise `LifecycleError` if they try.
- The pure research loop (`run_cycle`) still parks non-autonomy proposals at
  `AWAITING_USER_APPROVAL` with `acted_on_market=False` / `approved_anything=False`.
- No module in `research/auto_research/` imports an order / broker / kite / Telegram path
  (guarded by a test that scans the source).

## Honesty rails (inherited from the whole overhaul)

- **No research-grade data ⇒ no verdict.** The cycle concludes
  `Discovery unavailable — historical research data is not ready.` and proposes nothing.
  `canonical_readiness()` fails **closed** to red on any error.
- **Synthetic is never market evidence.** A synthetic evaluator (tests only) can produce a
  survivor, but it is never presented as evidence and never becomes a proposal.
- **Learning never mutates an active strategy.** Decay in a family yields a **new, re-tested
  child version** (`spec.bump_version` → new config hash; old evidence cannot transfer) — a
  proposal for the human, not an edit.

## Modules

| File | Role |
|------|------|
| `research/auto_research/thread.py` | append-only chain-of-thought log (JSONL, deterministic, torn-line-safe) |
| `research/auto_research/loop.py` | `run_cycle()` — the honest one-pass loop + `canonical_readiness()` |
| `research/auto_research/learning.py` | `LearningLedger` — per-family decay/improvement across cycles; proposes re-tested children |
| `research/auto_research/paper_book.py` | `PaperBook` — self-contained simulated ledger with the house risk caps |
| `research/auto_research/paper_autonomy.py` | `PaperAutonomyManager` — deploy → trade → retire losers, paper-only, live-locked |
| `research/auto_research/scheduler.py` | `AutoResearchBrain` — headless daemon + synchronous `run_once()` + paper-autonomy switches |
| `ui/auto_research_page.py` | "watch it think" panel + paper-autonomy toggle & paper book (More Tools → 🧠 Research Brain) |
| `tests/test_auto_research.py` · `tests/test_paper_autonomy.py` | 21 + 23 deterministic, network-free tests |

## Determinism

Same discovery seed + same injected evaluator ⇒ the same thread and the same proposals.
Wall-clock time is provenance only (the `stamp` field) and never part of any identity or the
content of a reasoning step. Production wires a real point-in-time backtest as `evaluate_fn`;
tests inject deterministic evaluators clearly labelled synthetic or real.
