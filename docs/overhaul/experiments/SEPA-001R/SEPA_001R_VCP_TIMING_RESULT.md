# SEPA-001R VCP timing result

## 1. Was the old detector late?

**Yes.** SEPA-001: median distance to the *pattern-high* pivot ≈ **+9.9%**, 72% of detections already outside the 1.5% buy-zone, E/F ≈ 0 fills.

## 2. Why?

Three stacked effects, not a too-tight zone:

1. Pivot = highest (usually earliest) contraction high, while price coils under later resistance.
2. `TOO_FAR_BELOW_PIVOT` (92%) **failed the pattern**, so the engine never tracked a live coil into the zone.
3. `sample_step=10` skipped 1–3 session windows.

Zigzag confirmation itself is causal (a swing exists only after `min_reversal_pct`). Fractal `find_swings(left,right)` is **not** on the eligibility path.

A fourth effect remains after the pivot change: after the last *confirmed* contraction high, price can run >2.5% without confirming a new swing. First daily snapshot of that base is often already `EXTENDED`. The lifecycle must wait for a later return to the last high — or refuse.

## 3. How much latency was removed?

- Planted VCPs: structure knowable **before** breakout; pivot knowable date ≥ extreme date.
- **MOTHERSON**: first new detection **−0.21%** vs old **−3.9%**.
- **TCS**: new detector sees a coil legacy missed.
- Unique-setup median distance at first snapshot is still **+10.7%** — now classified extended, not filled.
- First `ENTRY_READY` can be **months later** than first structural print (CHENNPETRO 2025-12-15 → 2026-03-20) without changing the zone.

## 4. What percentage are now in the intended entry region?

Of 944 unique setups at first snapshot: **7.3%** inside −0.25%…+1.5%. States: 711 `EXTENDED`, 164 `PIVOT_DEFINED`, 69 `ENTRY_READY`. Fully eligible (8/8 + zone + stop): **5** snapshots / **4** F fills.

## 5. Did earlier detection increase false positives?

Volume dry-up, tightening, base-depth, and Stage-2/RS gates are unchanged. Legacy-missed TCS was **below** the zone, not a chase. Extended prints are refused (126 E refusals).

## 6. Did the change improve executable trade frequency?

**Yes, from ~0 to a handful — not to a study.** SEPA-001 F≈0; SEPA-001R F=4 valid fills + 1 missed + 1 gap-through. Daily evaluation is what captured CHENNPETRO / LAURUSLABS / MOTHERSON / SBIN `ENTRY_READY` dates. That does **not** make core SEPA powered.
