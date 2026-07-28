# QuantTerm — The Simple Manual (printable)

*Written for an ordinary person with no trading, maths or programming background.
If you can read this, you can use QuantTerm safely.*

---

## 1. One-minute explanation
QuantTerm is a careful assistant for the Indian stock market. It finds *possible*
setups, checks them against strict rules, and tells you **why** it said yes, no or wait.
It can **practise with pretend money** and **study history** to test ideas. Real-money
trading is **switched off on purpose**. It never promises profit.

## 2. Ten-minute quick start
1. Open **Simple Home**. Read the big status line (Ready / Needs attention / etc.).
2. Do the single **Next best action** it shows.
3. Open **Getting Started** — a 7-step tour.
4. Do the **Practice Walkthrough** (a safe, made-up example — no real money).
5. Read one **skipped** setup and its plain reason.
6. Keep the **Glossary** open for any hard word.

## 3. The three modes
- **Research** — studies old data. Makes no trades. 
- **Paper** — practises with pretend money. Nothing reaches a real account.
- **Live** — real money, **locked** until strict safety + evidence rules are met.

## 4. Navigation map
- **TODAY:** Simple Home · Opportunities · My Positions · Alerts
- **LEARN & VERIFY:** Research Lab · What We Have Learned · Reports
- **SYSTEM:** Safety and Limits · Data Health · Settings · Help

Advanced Mode (a toggle) adds technical detail. It changes what you *see*, never what
the system can *do*.

## 5. Daily routine
1. Open Simple Home — ready or needs attention?
2. Check Data Health — must say healthy.
3. Confirm mode — should be *Paper practice*.
4. Read setups — check the plain **why**, the entry, the stop, and the **most you
   could lose**.
5. Let it record everything. Accept "no trade" when nothing qualifies.

> A good day is not necessarily a profitable day. A good day is one where the system followed its rules and protected the account.

## 6. When I do this, what happens?
| Action | Real money? | System checks | What changes | How to undo | Verify |
|---|---|---|---|---|---|
| Open the app | No | Data health, market status, current mode | Shows you the Home status | Just close it | Read the Home status line. |
| Refresh data | No | Whether fresh data can be loaded | Updates the prices shown | Nothing to undo | The data-health line should say 'healthy'. |
| Select PAPER mode | No | That paper mode is available | Marks decisions as pretend-money | Switch mode again | The mode indicator reads 'Paper practice'. |
| Arm PAPER autopilot | No | Gates, limits, safety stop | Allows automatic PAPER decisions | Disarm autopilot | Status shows 'Autopilot ON' and it's still PAPER. |
| Disarm autopilot | No | Nothing risky | Pauses automatic decisions | Arm again | Status shows 'Autopilot OFF'. |
| Accept a Telegram paper action | No | It is a paper action (always) | Records a PRETEND trade | Close the paper position | It appears as a paper position, not a live order. |
| Dismiss an alert | No | Nothing risky | Hides that alert | Alerts can reappear | The alert list updates. |
| Open a candidate explanation | No | Nothing risky | Shows why it qualified/was skipped | Just close it | You can read the main reason in plain words. |
| Open a paper position | No | Nothing risky | Shows entry, stop, max loss | Just close the view | You can see the maximum planned loss. |
| Close a paper position | No | Current paper price | Records a pretend result | It's recorded; you can't un-close history | It moves to your results. |
| Change a risk setting | No | New value is within safe bounds | Changes future PAPER sizing/limits | Change it back | The confirmation showed the exact before/after. |
| View a rejected setup | No | Nothing risky | Shows the skip reason | Just close it | You understand which rule it failed. |
| Open the Research Lab | No | Nothing risky | Shows past historical tests | Just leave the page | You see PASS/FAIL/INCONCLUSIVE, not orders. |
| Read PASS/FAIL/INCONCLUSIVE | No | Nothing risky | Nothing | n/a | PASS ≠ guaranteed; INCONCLUSIVE = not enough data. |
| Encounter DATA_UNAVAILABLE | No | That the data is missing | Shows an honest explanation | n/a | Follow the operator step to install data. |
| Encounter stale data | No | Data freshness | Labels prices as old | Refresh data | Prices are marked stale until refreshed. |
| Hit the daily safety stop | No | Day P&L vs the limit | Blocks new trades today | It resets next trading day | Status shows 'Daily safety stop active'. |
| Hit the trades-per-day limit | No | Trades used vs allowed | Blocks more trades today | It resets next trading day | Status shows trades used = allowed. |
| Attempt to access LIVE mode | No | The live migration lock | Nothing — live stays locked | n/a | You see the 'Live trading locked' message and why. |
| View broker-reconciliation status | No | Whether system and broker records match | Nothing | n/a | It shows matched/mismatch — a mismatch blocks anything live. |

## 7. PAPER walkthrough (safe, made-up)
1. **A possible setup appears** — Pretend stock 'PRACTICE LTD' has been quiet for weeks, then jumps above its resistance price on heavy volume. The system flags it as a possible setup.
2. **Why it qualified** — It was already a leader, its group was strong, the quiet base was tight, and the breakout closed above the required price with a small, sensible risk.
3. **Why another one was skipped** — Pretend stock 'CHASE LTD' also jumped — but it was already far above its normal trend. Buying that high is chasing, so the system skipped it.
4. **Entry is known only AFTER the signal** — The system waits for the breakout candle to close, then plans to enter on the NEXT day. It never pretends to buy at a price it couldn't have known.
5. **The structural stop** — The exit-if-wrong level sits just below the setup's structure. If price falls there, the practice trade closes to limit the loss.
6. **Your maximum possible loss** — Before anything, you can see the worst case: (entry − stop) × size. You always know the most you could lose on a practice trade before it starts.
7. **A no-fill day** — Sometimes the stock gaps far past the entry, so a real order couldn't have happened there. The system records 'no fill' instead of a pretend perfect entry.
8. **A winning outcome** — 'PRACTICE LTD' keeps rising and later closes below its trailing line. The practice trade exits with a pretend gain. It is recorded honestly.
9. **A losing outcome** — Another practice trade drops to its stop and exits for a pretend loss. Losses are normal — the rules keep each one small.
10. **The daily safety stop** — After a few pretend losses hit the day's limit, the system stops trading for the day. No 'making it back'. It resets next trading day.
11. **No valid opportunity** — On many days nothing qualifies. The system does nothing — and that is a good day, because it protected the account.
12. **Research: PASS, FAIL, INCONCLUSIVE** — Separately, the Research Lab tests ideas on history. PASS = promising (not guaranteed). FAIL = didn't meet the test. INCONCLUSIVE = not enough data.

## 8. Research Lab walkthrough
Open the Research Lab → pick a test → read the result:
- **PASS** = Promising under the registered test — NOT a promise of profit.
- **FAIL** = The idea did not meet the required test.
- **INCONCLUSIVE** = Not enough trustworthy evidence to say pass or fail.
- **DATA_UNAVAILABLE** = The required historical data is not installed, so the test cannot be judged.
The Lab makes **no trades**. It only studies the past.

## 9. Example: a setup that QUALIFIED
"Practice Ltd" was already a leader, its group was strong, it went quiet in a tight
range, then **closed** above its resistance on heavy volume with a small, sensible risk.
The system shows the entry (next day), the stop, and the maximum loss.

## 10. Example: a setup that was REJECTED
"Chase Ltd" also jumped — but it was already **too far above its normal trend**. Buying
that high is chasing, with little room before the exit. The system **skipped** it and
said so plainly.

## 11. The daily safety stop
After losses reach the day's limit, the system **stops for the day**. No "making it
back". It resets next trading day. If you see "Daily safety stop active", you are done.

## 12. Data health
Everything depends on good data. The system never shows a green "Ready" when data is
missing or stale. **Missing is unknown, not zero.** If research data isn't installed you
see **INCONCLUSIVE — DATA UNAVAILABLE**, with the operator step to fix it.

## 13. Telegram
Telegram is **paper-only**. A tap can only ever record a pretend trade. It **cannot**
place a real order — ever.

## 14. Troubleshooting
See `docs/user-guide/TROUBLESHOOTING.md`. Golden rule: open the page's "What is this
page?" panel or the Glossary — never guess.

## 15. Glossary
- **Autopilot armed** → Automatic paper decisions are permitted
- **Autopilot disarmed** → Automatic decisions are paused
- **Base** → A long, quiet price range before a breakout
- **Broker reconciliation** → Checking that system and broker records match
- **Candidate** → Possible setup
- **Circuit breaker** → Daily safety stop
- **Config hash** → Fingerprint of the rules
- **DATA_UNAVAILABLE** → Required historical data is not installed
- **Dataset snapshot** → Exact data version
- **Drawdown** → Fall from the previous account high
- **Eligibility** → Does it qualify?
- **Evidence Lab** → Research Lab
- **Expectancy** → Average result per trade
- **Experiment** → Structured historical test
- **FAIL** → Did not meet the required test
- **INCONCLUSIVE** → Not enough trustworthy evidence
- **Migration lock** → Temporary live-trading lock
- **No fill** → The assumed trade could not realistically happen
- **PASS** → Promising under the registered test, not guaranteed
- **Pivot** → The price the stock must break above
- **Point-in-time safe** → Uses only information available at that time
- **Provenance** → Record of where the result came from
- **Regime** → What kind of market we are in right now
- **Rejection reason** → Why it was skipped
- **Sector strength** → Whether the stock's group is doing well too
- **Slippage** → Difference between expected and actual price
- **Structural stop** → Exit level based on the setup's price structure
- **Trades-per-day limit** → Daily trade limit

## 16. What NOT to do
- **Don't** Treat every alert as 'buy now'.  
  *Why:* Alerts are things to look at, not commands. Many are skipped after a closer look.
- **Don't** Assume a high score means guaranteed profit.  
  *Why:* A score ranks setups; it does not predict the future. High-scoring setups still fail.
- **Don't** Switch to LIVE because a synthetic or practice test passed.  
  *Why:* Practice profit is not real evidence. Live needs formal, measured proof.
- **Don't** Change lots of settings after a bad run.  
  *Why:* Tweaking to fit past losses usually makes the next result worse, not better.
- **Don't** Bypass the daily safety stop.  
  *Why:* The stop exists to end bad days early. Trading through it is how small losses become big ones.
- **Don't** Keep re-arming after a risk lock.  
  *Why:* The lock is telling you today is done. Forcing more trades fights your own safety net.
- **Don't** Treat paper profit as proof you'll profit for real.  
  *Why:* Real trading adds slippage, gaps and emotion that paper does not.
- **Don't** Assume missing data means zero.  
  *Why:* Missing is unknown, not zero. Zero would quietly corrupt every result.
- **Don't** Buy just because a price moved or 'looks cheap/expensive'.  
  *Why:* Price and valuation alone are not the system's setup. That's guessing.
- **Don't** Rely on Telegram as a live-order channel.  
  *Why:* Telegram is paper-only, by design. It can never send a real order.
- **Don't** Expect a trade every day.  
  *Why:* Most days have no valid setup. Forcing trades is the fastest way to lose.
- **Don't** Use money you cannot afford to lose.  
  *Why:* Even a good process has losing streaks. Only risk what won't hurt your life.

## 17. Success checklist (success = following the process)
1. Confirm the system says data is healthy.
2. Confirm which mode you are in (Research / Paper / Live).
3. Begin in PAPER mode.
4. Read WHY a setup qualified or was skipped.
5. Check the entry, the stop, and the maximum possible loss.
6. Do not override safety limits.
7. Let the system record every outcome.
8. Review results over a meaningful number of trades, not one.
9. Accept 'no trade' when nothing valid exists.
10. Move towards LIVE only after formal evidence and deployment sign-off exist.

## 18. Frequently asked questions
- **Will this make me money?** No one can promise that. It aims to follow a careful
  process and protect your account.
- **Why did it not trade today?** Most days nothing qualifies. "No trade" is often the
  right answer.
- **A synthetic/practice test PASSed — can I go live?** No. Practice is not real
  evidence. Live needs formal proof and sign-off.
- **It says DATA_UNAVAILABLE — is it broken?** No. It is being honest that the data for a
  test isn't installed, so it won't guess.
- **Can Telegram place a real order?** No, never. Paper only.
- **Is a high score a guarantee?** No. A score ranks a setup; it can't see the future.

## 19. What QuantTerm cannot promise
- It cannot promise profit — the market is uncertain.
- A PASS in research is not a promise of future money.
- Paper (practice) profit is not proof of real profit.
- It will not produce a trade every day — most days there is nothing valid.
- It cannot protect you from risking money you can't afford to lose.
