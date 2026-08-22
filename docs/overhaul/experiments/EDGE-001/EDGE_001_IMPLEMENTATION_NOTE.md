# EDGE-001 — Implementation Note (audit before strategy code)

**Experiment:** EDGE-001 — NSE Cross-Sectional Momentum  
**Independence:** not SEPA, not FEATURE-001/002, not production scanner tuning.  
**Written before** momentum portfolio code.

FEATURE-002 remains frozen. This note does not change Trend/RS shadow ranks, candidate logging, graduation criteria, or production BUY.

---

## 1. What already exists (reuse)

| Piece | Path | EDGE-001 use |
|---|---|---|
| Official bhav OHLCV | `data/bhavcopy_store.get_ohlcv` via `load_store_frames` | Primary prices. CA applied **on read**; store stays raw |
| PIT investable screen | `research/sepa/universe_pit.py` `FastInvestable.snapshot` | Rebalance-date universe: min price 20, min 20d turnover ₹50L, min 260 sessions, no future listings |
| `rs_cs_v1` | `research/sepa/rs.py` + `FastRS` | Comparative ranker **M3** only. Weights frozen |
| CNC cost model | `core/costs.py` `round_trip_cost_pct("CNC")` = 0.22% fees + 0.10% slippage = **0.32%** round-trip | Sole cost model. Gross and net both reported |
| Regime (descriptive) | `research/sepa003/regime.py` `classify_regime_level` | H4 breakout only. **Not a gate** |
| Official Nifty | `data/index_store.build_from_local` | Benchmark if local CSVs exist; never yfinance |
| Harness | `research/harness.py` PSR / DSR / block-bootstrap / BH-FDR | Monthly portfolio return inference |
| Feature store | `research/feature_store.py` | Not used (separate EDGE-001 ledger) |
| Production `MOMENTUM` | `UnifiedScanner._analyze`: 5-day return + RSI 50–70 + volume, **time-series** | Comparison only. Do not change |
| EXP-006 | Institutional momentum **breakout** (base + breakout + stop) | Different hypothesis. Blocked/incomplete. Do not merge |

There is **no EXP-003** momentum CS study in `docs/overhaul/experiments/`. FEATURE-001 used `rs_cs_v1` as a **rank feature on scanner fires**, not as a monthly long-only portfolio. That history is consumed for FEATURE-001 questions; EDGE-001’s object is different (full investable cross-section, scheduled rebalance). Still: 2019–2026 is not philosophically pristine.

---

## 2. What EDGE-001 will not do

- Touch `research/feature002/**` or `scan/auto_scan.py` observe hook
- Change production BUY, Ready, autopilot, GTT, broker, Telegram
- Add ATR/5%/trail stops or profit targets (that would be a later EDGE-002)
- AI, news, DeepSeek, conviction, VCP, SEPA Core F
- Optimize Top-N or rebalance frequency after seeing confirmation
- Introduce a regime gate after seeing H4

---

## 3. Execution convention (locked)

1. On rebalance session **T** (last official session of the calendar month), rank using closes **≤ T**.
2. Trade at the **next session open T+1** (no same-close fill).
3. Exit at the next rebalance’s executable open (T_next+1).
4. Equal weight, long only, no leverage, no pyramiding.
5. If a name has no next open, drop it (do not invent a fill).

---

## 4. PIT limitations (state honestly)

- Membership is inferred from bars present ≤ T (`bhav_inferred`) unless a point-in-time listing file exists. **PIT_DEGRADED** for listing identity.
- Sector map is today’s `nse_universe` grouping applied historically → **PIT_DEGRADED** for sector concentration (descriptive).
- CA: adjustment-on-read. Unresolved split gaps can remain if `logs/ca_events.json` is thin.
- Nifty 500 official series may be absent; then the broad benchmark is **equal-weight investable universe**, not NIFTY 500.
- FEATURE-001/SEPA already mined 2019–2026. Confirmation 2025–2026 is a **held-out block for this protocol**, not an untouched lifetime OOS.

---

## 5. Production MOMENTUM vs EDGE-001

Production `MOMENTUM` is short-horizon **time-series** (5-day %, RSI, volume surge) attached to a 2×ATR stop/target ticket. EDGE-001 is **12-1 cross-sectional** rank, monthly, no stop. They can correlate without being the same phenomenon. Comparison uses the same next-open monthly hold so expectancy is not mixed with the scanner’s 10–20 day sim.
