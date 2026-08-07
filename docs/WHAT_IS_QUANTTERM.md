# What Is QuantTerm? (Read this first — assumes you know nothing)

## The one-line version

**QuantTerm is a system that hunts for a real, provable edge in the Indian stock
market — and, just as importantly, refuses to fool itself into believing a fake
one.**

It is not a "buy-and-hold index" app and it is not a signal-selling bot. It is a
disciplined research organisation and a survival-first execution engine, built for
one purpose: to find out whether a genuine trading edge exists, prove it to a
scientific standard, and only then deploy real money behind it — safely.

---

## The problem it exists to solve

Everyone has trade ideas. Breakouts, patterns, momentum, tips — ideas are cheap and
endless. **The hard part isn't the idea. It's knowing whether the idea actually
makes money, or just *looked* good by luck.**

Three brutal facts:

1. **93% of Indian traders lose money** (SEBI, 2024 — ₹1.8 lakh crore lost in three
   years). Not because they lack ideas — because they cannot tell a real edge from a
   mirage, and they bet real money on false confidence.
2. **Backtests lie.** A strategy can look brilliant on historical data and be
   completely fake — because of survivorship bias (only today's winners are in the
   data), overfitting (you tried 200 things and kept the luckiest), un-adjusted
   prices (splits look like crashes), or simply ignoring trading costs.
3. **Almost all trading software helps you trade *more*.** Almost none helps you know
   whether you *should*.

QuantTerm is built to answer one question, honestly, before a single rupee is
risked: **"Does this actually work — with real money, after costs — or am I fooling
myself?"**

---

## The philosophy (two rules everything obeys)

1. **Survival first.** You cannot win if you blow up. Risk is capped *before* returns
   are ever chased — 1% per trade, hard limits, an exchange-side stop on every
   position.
2. **Evidence over vibes.** No claim without proof. A gorgeous-looking idea on thin
   or biased data is treated as *guilty until proven innocent*. "We tried it and it
   went up" is not evidence; a statistical survival test is.

Everything else in the system is a consequence of these two rules.

---

## What the system actually does (the pipeline, in plain words)

Think of it as an assembly line: raw data goes in one end, a disciplined decision
comes out the other, and a scientific court sits in the middle checking that nobody
is lying.

- **Data layer — get honest data.** Pulls real NSE market data, refuses to invent
  anything, adjusts for splits/bonuses so old prices aren't fake crashes, and flags
  stale data loudly. Garbage in = confident lies out, so this layer is paranoid.
- **Signal layer — the idea generator.** Scans the *entire* market every few minutes
  for setups: breakouts, momentum, chart patterns, accumulation. This is the easy,
  cheap part — ideas.
- **Risk layer — the gatekeepers.** Decides *how much*, if anything, to bet: position
  sizing, total portfolio risk, and correlation (are your five "different" bets
  actually the same bet?). This is where survival is enforced.
- **Execution layer — trade with a seatbelt.** Every live trade ships with a
  stop-loss placed *at the exchange*, so you are never holding a naked, unprotected
  position — even if your laptop dies.
- **The Brain — one honest read.** Composes regime, edge health, risk, and news into
  a single posture (aggressive / normal / defensive / stand-aside) and a short list
  of directives. Read-only; it advises, it doesn't gamble.
- **Alerts — it talks to you.** Proactive Telegram messages so you don't have to
  stare at a screen.

That is a competent trading terminal. But the part that makes QuantTerm *different*
is what sits in the middle.

---

## The part that makes it different: the science

Most systems stop at "here's a signal." QuantTerm adds a **courtroom** that puts
every strategy on trial before it is believed.

- **The Gauntlet (`python -m gauntlet`).** A single command that runs a strategy
  through the full statistical battery — Deflated Sharpe (kills "we tried many
  things"), White's Reality Check (is the best signal genuinely good or just the
  luckiest?), False-Discovery-Rate correction, alpha-vs-beta (is it skill, or just
  market exposure?), correlation-aware confidence intervals, and a regime split —
  **all net of realistic costs.** It returns exactly one verdict per strategy:
  **PASS / FAIL / INCONCLUSIVE.** No spin.
- **Evidence Levels (E0–E6).** Every capability is rated by how much *real* proof
  backs it — from "designed" (E0) to "stable across market regimes" (E6). A level
  can only rise through an objective, gated test, never by opinion. Today the
  strategy's edge sits at **E0 — unproven.** The system will not pretend otherwise.
- **Governance — it stops itself.** Kill conditions, rollback triggers, a human kill
  switch. If data looks corrupt or losses breach a limit, it halts new live orders
  automatically.
- **The Research Log — a lab notebook.** Every experiment is *pre-registered* (the
  rules fixed before the run) and recorded — wins **and** losses. No cherry-picking,
  no quietly re-running until something passes. Negative evidence is treated as
  valuable as positive.

This is the whole point: **the system is built to catch its own lies.**

---

## What we've done so far (the honest journey)

We built the full stack above, then actually *ran the science* — five pre-registered
experiments (all in `RESEARCH_LOG.md`):

- **EXP-002 — the 16 short-term breakout/pattern signals.** Verdict: **FAIL.** After
  costs, essentially all of them were net-negative. (This matches SEBI's 93%.)
- **EXP-003 / EXP-004 — momentum, then risk-managed momentum.** Verdict:
  **INCONCLUSIVE / FAIL.** Momentum showed a *real* positive edge and beat the index
  on raw return — but not on a risk-adjusted basis (it carried far more volatility
  and a 44% drawdown), and a 200-DMA trend filter reduced the risk without closing
  the gap to the index.
- **EXP-005 — momentum over ~15 years.** Verdict: a spectacular-looking **PASS**
  (37% CAGR, Sharpe 1.48) — **which the system correctly rejected as INVALID.** The
  long-history data was survivorship-biased (only stocks that survived 15 years),
  which inflates results into a mirage. The system caught the illusion instead of
  believing it.

**Current honest status:** no *proven* edge on trustworthy data yet. Therefore: **no
real money at risk.** The edge's Evidence Level is **E0**.

---

## Why that is a success, not a failure

The system did *exactly its job*: it stopped us from betting real money on ideas that
don't hold up — including one that looked like a 37%-a-year fortune. **That is the
entire difference between the 7% who survive and the 93% who don't.**

The goal was never "find a strategy no matter what." The goal was "deploy only a
*real* one, and never fool ourselves." We are honouring that. A rigorous "not yet" is
worth more than a confident lie.

---

## What this system is ultimately FOR

- It is an **edge-discovery-and-proof machine** bolted onto a **survival-first
  execution engine.**
- Its standing job: keep rigorously hunting for a real, durable edge — and the moment
  one **PASSES the gauntlet on clean data**, deploy it safely, sized correctly, fully
  monitored, with the governance layer watching. Until that day, it protects capital
  by refusing to pretend.
- The index is not the goal; it is merely the **current benchmark that nothing has
  beaten yet.** The mission is to find something that genuinely does — or to keep
  saying "not yet," honestly, for as long as the evidence demands.

---

## What's next

The single real bottleneck to giving the edge search a fair shot is **clean,
survivorship-free, long-history data** (including delisted names, corporate-action
adjusted). That is the frontier — free sources are either too short (clean NSE
bhavcopy, ~7 years) or biased (yfinance, 15 years but survivors-only). A paid,
survivorship-free dataset is what would let the gauntlet deliver a truly definitive
verdict on momentum or any other candidate.

In the meantime: run the app in paper mode to accumulate genuine forward evidence,
and put every new idea through the same discipline — **pre-register → gauntlet →
only then, real money.**

---

_QuantTerm's promise is not riches. It is the truth about whether an edge exists —
and the discipline to act on that truth instead of a mirage._
