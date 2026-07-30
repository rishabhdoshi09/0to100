# How QuantTerm gets smarter each day

> "It is an infant who is growing up each day; getting smarter each day."

This is the honest, coded answer — not a slogan. Every mechanism below is real code in
`research/auto_research/` with deterministic tests in `tests/test_growth.py`.

## The one idea that matters

**A good backtest is not an edge. An edge is a backtest that keeps working on data it has
never seen.** So the system does BOTH, automatically, and compares them. That comparison is
where the learning comes from.

## The daily loop (`AutoResearchBrain.grow_one_day`)

Once you engage paper autonomy, the brain runs this every day, hands-off:

```
1. STRATEGIZE   discovery invents readable candidate strategies from the approved grammar,
                biased by what has forward-tested well so far (see "adaptive search" below).

2. BACKTEST     each candidate is evaluated on real historical bhavcopy data (in-sample).
                Only survivors of the full evidence gate go forward. (research/momentum_breakout)

3. FORWARD TEST survivors are auto-deployed to PAPER and traded each day on the LATEST bars —
                data the backtest never saw. This is out-of-sample, in real time.
                (paper_book.py + paper_autonomy.py)

4. CALIBRATE    compare each strategy's forward (paper) edge to its backtest edge:
                  • forward keeps ≥70% of backtest R  → CONFIRMED  (the edge is real)
                  • forward keeps <30%                → DECAYED    (fading)
                  • forward edge ≤ 0                  → OVERFIT    (the backtest lied)
                (growth.py — calibrate())

5. OBSERVE + IMPROVE
                CONFIRMED strategies keep trading; DECAYED/OVERFIT ones are retired
                autonomously. The verdict is folded into persistent memory, which raises or
                lowers TRUST in that whole family of ideas. (knowledge.py)

6. REMEMBER     memory is saved to disk, so a restart doesn't reset the child to zero. The
                learning compounds day over day.
```

## The three concrete "getting smarter" mechanisms

### 1. Forward-test calibration (catching overfits) — `growth.calibrate()`
The single most valuable skill in quant is telling a real edge from a lucky/overfit backtest.
The system does it by measuring how much of the backtested edge actually survives forward in
paper. An idea that looked great in-sample but bleeds in paper is caught and stood down — on
its own, with no human needed.

### 2. Persistent trust that shifts the search — `knowledge.py`
Each strategy family carries a **trust** score in [0,1]. Forward-confirmation pushes it up;
overfit pushes it down. Trust is saved to `logs/auto_research/knowledge.json`, so the system
literally remembers yesterday's lessons.

### 3. Adaptive discovery — `discovery.generate(family_weights=…)`
Tomorrow's search is drawn proportional to trust: families that keep working out-of-sample
get **more** attempts, chronic decayers get **fewer** — but never zero, so it keeps exploring
(an infant still tries new things). Over weeks, the search itself concentrates on what the
market has actually rewarded. That is the compounding "smarter each day."

## What stays bolted down (unchanged)

Growth is **paper-only and live-locked.** It trades simulated money and can never reach live:
the `paper_autopilot` actor is structurally barred from the only lifecycle transition toward
live review (`PAPER_EVALUATION → ELIGIBLE_FOR_LIVE_REVIEW` stays user-only). It still refuses
SYNTHETIC evidence and a red data gate, so with no real market data it grows *nothing* and
says so honestly — it never fabricates a backtest or a price to look busy.

## Where to watch it

- **🛰️ Control Room** → "What it has learned" shows trust per family (backtest R vs forward R).
- **🧠 Research Brain** → "Grow one day now" runs a full day on demand; the thread shows every
  calibration verdict as it happens.
