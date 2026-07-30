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

## The one boundary (a deliberate seatbelt)

The brain drives research all the way to `AWAITING_USER_APPROVAL` on its own and **stops**.
It never approves a strategy, never activates paper, never touches live, never places an
order. By the project's own safety directives, real-money autonomy requires a person's
approval — which happens in **Strategy Studio**, not here, and never automatically.

This boundary is **structural**, not a promise:
- `loop._advance_to_gate` walks the lifecycle using only `actor="system"` transitions; the
  step beyond the gate (`AWAITING_USER_APPROVAL → APPROVED_FOR_PAPER`) is `actor="user"` and
  raises `LifecycleError` for the system.
- Every `CycleReport` carries `acted_on_market=False` and `approved_anything=False`.
- No module in `research/auto_research/` imports an order/broker/Telegram path.

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
| `research/auto_research/scheduler.py` | `AutoResearchBrain` — headless daemon + synchronous `run_once()` |
| `ui/auto_research_page.py` | read-only "watch it think" panel (More Tools → 🧠 Research Brain) |
| `tests/test_auto_research.py` | 21 deterministic, network-free tests |

## Determinism

Same discovery seed + same injected evaluator ⇒ the same thread and the same proposals.
Wall-clock time is provenance only (the `stamp` field) and never part of any identity or the
content of a reasoning step. Production wires a real point-in-time backtest as `evaluate_fn`;
tests inject deterministic evaluators clearly labelled synthetic or real.
