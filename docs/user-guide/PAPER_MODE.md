# Paper mode (safe practice)

**Paper mode practises with imaginary money. No real money is ever sent to a broker.**
This is where you learn.

## How a practice trade works
1. A possible setup appears (a stock breaks above a price it struggled to pass).
2. The system checks the rules and shows you **why** it qualified.
3. Entry is planned for the **next** day — never at a price it couldn't have known.
4. A **structural stop** (exit-if-wrong) sits just below the setup's structure.
5. You can see the **maximum possible loss** before anything: (entry − stop) × size.
6. The trade is recorded honestly — win, loss, or "no fill".

## "No fill" — why a trade sometimes doesn't happen
If the stock jumps far past the planned entry, a real order could not have filled there.
The system records **no fill** instead of pretending. This keeps practice honest.

## Autopilot
- **Armed** = automatic *paper* decisions are allowed.
- **Disarmed** = automatic decisions are paused.
- Arming changes nothing about real money — paper stays paper.

## Telegram
Telegram can only ever trigger a **paper** action. It **cannot** place a real order —
ever, by design. (See `TELEGRAM.md`.)

## The honest truth about paper profit
Practice profit is **not** proof of real profit. Real trading adds slippage, gaps and
emotion that practice does not.

👉 Next: `RESEARCH_LAB.md`
