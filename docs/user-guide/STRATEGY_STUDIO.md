# How to Create and Improve a Strategy

A plain-language guide to **Strategy Studio** (More Tools → 🧪 Strategy Studio). The
Studio can suggest its own trading ideas, test them honestly, and show you the case
**for and against** each one. **It can never place a trade.** Only *you* can approve an
idea for paper practice; real-money (LIVE) trading stays locked.

## 1. What a strategy is
A strategy is a simple set of rules: **when to buy**, **where you're wrong (the stop)**,
and **when to exit**. Nothing more magical than that.

## 2. Entry, stop and exit
- **Entry** — the condition that makes a stock interesting (e.g. it breaks above a price
  it struggled to pass). Entry always happens the **next day**, never at a price the
  system couldn't have known.
- **Stop** — the level that proves the idea wrong. If price falls there, you're out.
- **Exit** — how you let winners run and then leave (e.g. when the trend breaks).

## 3. How QuantTerm creates strategies
It combines a small set of **approved, safe building blocks** (returns, trend, breakouts,
volume, sector strength, market state…) into readable rules — within a strict budget, and
it **records every idea it tries, including the ones it rejects.** It does not blindly try
millions of combinations.

## 4. What historical testing means
Each idea is checked against **real past data**: would it actually have made money, after
costs? Without real data the Studio says *"Discovery unavailable — historical research
data is not ready"* and only shows a labelled **demonstration** — which is **not**
evidence.

## 5. Why costs matter
Every trade pays brokerage, taxes and slippage. Many ideas look good *before* costs and
lose *after* them. The Studio always subtracts realistic costs.

## 6. Why more filters are not always better
Adding rules can make a backtest look great by accidentally fitting the past. A **simpler**
idea that does about as well is usually **more trustworthy**.

## 7. What overfitting means (simple example)
If you write rules that perfectly explain last year — "buy only on Tuesdays after a gap in
March" — it may look perfect on history and fail completely next year. That's overfitting:
memorising the past instead of finding a real pattern.

## 8. PASS, FAIL, INCONCLUSIVE
- **PASS** — promising under the honest test (**not** a promise of profit).
- **FAIL** — did not meet the test (useful — it saves your money).
- **INCONCLUSIVE** — not enough trustworthy evidence to say.

## 9. How to read evidence confidence
The Studio shows **five separate** measures, never one number: how strong the *evidence*
is, how good the *data* is, how *stable* the idea was across periods/sectors, how well it
should *reproduce* in real trading, and (for models) a per-opportunity *prediction*. A
strategy is **not** trustworthy just because one of these is high.

## 10. How to inspect losing periods
Open **Convince Me → Why it might fail**: worst period, worst drawdown, longest losing
streak, the market conditions where it struggled, and what could invalidate it.

## 11. Tweak one rule at a time
In **Tweak**, ask in plain words ("reduce the maximum stop to 5%", "avoid weak markets").
Change **one thing at a time** so you can see its effect.

## 12. Why every material tweak needs a new test
Changing a rule changes the strategy — so the old evidence **no longer applies.** Every
material tweak creates a **new version** and must be **re-tested from scratch.** The
original version is preserved.

## 13. How to compare versions
In **Compare**, see complexity, trades, expectancy, drawdown, costs and stability side by
side. If two versions are within noise, the Studio says so and **never auto-picks the
higher number** — prefer the simpler, steadier one.

## 14. When to reject a strategy
Reject when: costs eat the edge, it depends on a few lucky trades or one period, it's too
concentrated, or the evidence is INCONCLUSIVE. A clean rejection is progress.

## 15. When a strategy is suitable for paper testing
Only when the evidence is **real** (not a demo), the data gate is **green**, and it
survives the honest tests. Then *you* — not the system — may **Approve for Paper Testing**.

## 16. Why paper success still does not permit live trading
Paper (pretend-money) success is encouraging but **not** proof of real profit. LIVE
trading needs separate, stronger evidence and a deployment approval that is **out of scope
here** — the live lock stays on.

---
Contextual help sits beside every major control, and a guided first-use walkthrough is
built into the page.
