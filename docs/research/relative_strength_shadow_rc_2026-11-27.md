# Relative-strength shadow release candidate

## Approval and registration

Proposed first decision session is **1 December 2026**, after the official NSE bhavcopy is available. Simulated entry is the next official session open. The date is proposed and the runtime registration is absent. If approval or host readiness misses the date, choose another prospective session and timestamp the registration before it.

The existing supervisor enqueues the observer as a separate noncritical job after the official EOD archive arrives. Transient missing files retry through the durable scheduler at five-minute intervals, up to twelve attempts. Integrity failures remain visible as failed jobs. Paper settlement and forward learning proceed independently. Without an approval record it returns DISABLED. The record is logs/product/relative_strength_shadow_registration.json under the approved runtime root. Set mode to SHADOW_ONLY and provide first_decision_session, an actual timezone-aware registered_at, spec_sha256, and implementation_sha256. Obtain the implementation hash with product.relative_strength_shadow.implementation_sha() from the approved checkout. The ledger binds the record hash and rejects later edits.

**Frozen strategy SHA-256** 3ad2d8db51b3afb3a456cf4d2ed2a601f0d8da86cc45075c0145b5555f9804f8. The observer reuses the discovery selector, scoped NSE validation, and common daily-bar paper simulator. It records a separate five-slot shadow portfolio and controls with no order routing, PaperBook mutation, or BUY promotion. The equal-weight eligible basket and seeded random draw have the same intended allocation and decision dates. All three use the same opening fill, 3% stop, 5% target, 2% breakeven trail, ten-session horizon, and 0.32% round-trip costs.

Each decision and both controls freeze before the next calendar day’s 09:15 IST deadline with a 30-minute buffer. A skipped freeze is recorded as missed. Missing whole-market files leave settlement pending. Unresolved holdings receive the existing conservative zero-recovery portfolio stress and are excluded from resolved-fill expectancy. The observer checks payload and archive identities and refuses a revised decision archive at settlement.

## Predefined evaluation

- Read the first 24 scheduled ten-session cycles. Require at least 20 matured cycles and valid overlap-aware inference before a verdict.
- Require a positive one-sided 95% lower bound for paired net portfolio return per cycle against the matched eligible basket, positive resolved-fill net expectancy, and positive aggregate portfolio return.
- Report matched random excess, daily mark-to-market drawdown, exit frequencies, and multiclass Brier score against the frozen historical forecast. The strategy-level target, stop, and expiry estimates are 27.7%, 63.1%, and 9.2% from 69 resolved fills with a joint half-count prior. The earlier seven-cycle fit scored 0.652 on the later seven discovery cycles.
- Verify upcoming NSE calendar coverage before crossing into a new year. Keep the specification and forecast fixed. A stale or missed registration cannot be repaired with a retrospective freeze.

## Historical sensitivity

The replay still ends October 31, 2025, preserving the November 2025 reserve. Starting on October 1, 2024 adds 14,331 admitted stock-session bars and gives 21 cycles on the same cohort and rules. The start change also shifts the cycle phase. The official November 1 special session is included. [2024 NSE calendar](https://nsearchives.nseindia.com/content/circulars/CMTR59722.pdf) · [May amendment](https://nsearchives.nseindia.com/content/circulars/CMTR61518.pdf) · [November amendment](https://nsearchives.nseindia.com/content/circulars/CMTR64960.pdf) · [Muhurat session](https://nsearchives.nseindia.com/content/circulars/CMTR64628.pdf)

| Selection | Original net per fill at 32 bp | 42 bp | 52 bp | 72 bp | Extended net at 32 bp |
|---|---:|---:|---:|---:|---:|
| Relative strength | +0.106% | +0.006% | −0.094% | −0.294% | −0.905% |
| Ascending triangle + pocket pivot | +0.305% | +0.205% | +0.105% | −0.095% | −0.315% |
| Ascending triangle + pre-breakout | +0.204% | +0.104% | +0.004% | −0.196% | −0.747% |

The cost sensitivity adds fixed per-fill drag while preserving fills and exits. It gives no estimate of slippage-induced fill changes. The extended relative-strength replay covers 105 resolved fills and trails the matched basket by 0.172 percentage points per cycle. The combination results cover 26 and 27 resolved fills in the extension and remain exploratory.

## Deployment preflight

- Verify the approved source SHA and frozen strategy digest. Confirm the protected existing paper lane and live interlock.
- After authorization, run the existing host preflight on the deployment target with market probes and a persistent runtime. The isolated no-network check was BLOCKED on unverified market access.
- Verify fresh official EOD archives, an adequate verified calendar, the existing forward learning schedule and SHADOW_ONLY mode before installing the approval record.
- Removing the approval record disables observation. Preserve any started ledger and its approval record for audit. A separate reviewer sign-off and actual forward observations remain outstanding.
