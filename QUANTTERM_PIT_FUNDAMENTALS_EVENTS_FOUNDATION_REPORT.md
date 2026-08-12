# QuantTerm PIT Fundamentals + Earnings/Announcements Foundation

> Data-foundation cycle only. No strategies run. No AI/ML. Production unchanged.
> Global trust remains `OPERATIONAL_ONLY`. Closed hypothesis branches stay closed.

## WHAT WE BUILT

QuantTerm can now remember **when** company results and announcements became public, and store statement metrics tied to that public time — not scrape time.

## WHAT IT MEANS

New economic ideas that need fundamentals or earnings dates can be tested honestly on the certified scope — without pretending today's screener cache is historical truth.

## WHAT QUANTTERM WILL DO

Keep production trading unchanged. Do not auto-trade on these datasets. Use them only for future preregistered research experiments.

---

## 1. Starting state

- Fundamentals/events were `OPERATIONAL_ONLY` / `NOT_PIT_SAFE` (as-of-now caches)
- PitContract refused fundamentals; valuations incomplete without `available_ts`
- Parent OHLCV scoped snapshot: `2f683be0c73eaa33` (870 CERTIFIABLE names)

## 2. Architecture (reuse, no parallel store)

| Piece | Path |
|-------|------|
| Events ledger | `data/pit_events.py` → `logs/pit_events.json` |
| Fundamentals ledger | `data/pit_fundamentals.py` → `logs/pit_fundamentals.json` |
| Valuations ledger | existing `data/pit_valuations.py` (derived PE) |
| NSE results ingest | `data/nse_results_ingest.py` |
| NSE announcements ingest | `data/nse_announcements_ingest.py` |
| PitContract domains | `fundamentals`, `events`, `valuations` |
| Immutable snapshot | existing `SnapshotStore` |

## 3. AVAILABLE_AT contract

- Earnings/results: NSE `broadCastDate` / `exchdisstime` / `filingDate`
- Announcements: NSE `an_dt` / `exchdisstime` / `sort_date`
- Fundamentals metrics: same broadcast time as the linked result XBRL
- **Forbidden:** mapping screener `fetched_at` → `available_at`

## 4. Materialization results

- Events: **43569** rows / **868** symbols
- Events by type: `{'EARNINGS_RESULT': 38234, 'FINANCIAL_RESULT_UPDATE': 5335}`
- Events date range: `['2020-01-02', '2026-08-11']`
- Fundamentals: **12584** rows / **860** symbols
- Fundamentals date range: `['2022-04-08', '2025-03-18']`
- Valuations (derived PE): **10427** rows / **813** symbols
- XBRL parse stats: `{'available': True, 'path': '/workspace/logs/pit_fundamentals.json', 'rows': 12584, 'symbols': 860, 'research_grade': True, 'source': 'nse_xbrl_financial_results', 'note': '', 'generated_at': '2026-08-11T18:05:55.296937+00:00', 'date_range': ['2022-04-08', '2025-03-18'], 'key_requirement': 'AVAILABLE_AT', 'metric_coverage': {'revenue_from_operations': 12162, 'other_income': 12584, 'profit_before_tax': 12162, 'profit_after_tax': 12160, 'comprehensive_income': 12160, 'basic_eps': 12162, 'diluted_eps': 12162, 'face_value': 12583, 'paid_up_equity_capital': 12583, 'debt_equity_ratio': 3766}, 'xbrl_attempted': 12635, 'xbrl_parsed': 12584, 'xbrl_errors': 51}`

## 5. Validation

- Overall ok: **True**
- Validator: `pit_fundamentals_events_foundation.v1`

```json
[
  {
    "check": "events_ledger_present",
    "ok": true,
    "detail": {
      "available": true,
      "path": "/workspace/logs/pit_events.json",
      "rows": 43569,
      "symbols": 868,
      "research_grade": true,
      "source": "nse_corporate_announcements:01-01-2025:11-08-2026",
      "note": "",
      "generated_at": "2026-08-11T18:01:39.928038+00:00",
      "by_event_type": {
        "EARNINGS_RESULT": 38234,
        "FINANCIAL_RESULT_UPDATE": 5335
      },
      "date_range": [
        "2020-01-02",
        "2026-08-11"
      ],
      "key_requirement": "AVAILABLE_AT"
    }
  },
  {
    "check": "events_have_available_at",
    "ok": true,
    "detail": "rows=43569"
  },
  {
    "check": "events_not_sample",
    "ok": true,
    "detail": "source=nse_corporate_announcements:01-01-2025:11-08-2026"
  },
  {
    "check": "fundamentals_ledger_present",
    "ok": true,
    "detail": {
      "available": true,
      "path": "/workspace/logs/pit_fundamentals.json",
      "rows": 12584,
      "symbols": 860,
      "research_grade": true,
      "source": "nse_xbrl_financial_results",
      "note": "",
      "generated_at": "2026-08-11T18:05:55.296937+00:00",
      "date_range": [
        "2022-04-08",
        "2025-03-18"
      ],
      "key_requirement": "AVAILABLE_AT",
      "metric_coverage": {
        "revenue_from_operations": 12162,
        "other_income": 12584,
        "profit_before_tax": 12162,
        "profit_after_tax": 12160,
        "comprehensive_income": 12160,
        "basic_eps": 12162,
        "diluted_eps": 12162,
        "face_value": 12583,
        "paid_up_equity_capital": 12583,
        "debt_equity_ratio": 3766
      }
    }
  },
  {
    "check": "fundamentals_have_metrics",
    "ok": true,
    "detail": "rows=12584 symbols=860"
  },
  {
    "check": "valuations_derived_or_present",
    "ok": true,
    "detail": {
      "available": true,
      "path": "/workspace/logs/pit_valuations.json",
      "rows": 10427,
      "symbols": 813,
      "research_grade": true,
      "source": "derived_nse_xbrl_eps_x_certified_close",
      "note": "",
      "generated_at": "2026-08-11T18:05:58.053650+00:00"
    }
  },
  {
    "check": "pit_fundamentals_domain",
    "ok": true,
    "detail": {
      "status": "READY",
      "reasons": [],
      "usable": true
    }
  },
  {
    "check": "pit_events_domain",
    "ok": true,
    "detail": {
      "status": "READY",
      "n": 46
    }
  },
  {
    "check": "pit_no_lookahead_fundamentals",
    "ok": true,
    "detail": "2024-04-22"
  },
  {
    "check": "never_fetched_at_as_available_at",
    "ok": true,
    "detail": "ledger validators refuse fetched_at-only rows"
  }
]
```

## 6. Immutable foundation snapshot

- Snapshot ID: **`46ff79f58ee21c9e`**
- Verify: `True`
- Parent OHLCV: `2f683be0c73eaa33`
- Scoped certification: `SCOPED_PIT_FUNDAMENTALS_EVENTS_READY`
- Global trust: `OPERATIONAL_ONLY` (not upgraded)

## 7. Newly testable hypothesis families

### READY_TO_TEST

- **post_earnings_drift_timing** — Official NSE earnings-result events carry AVAILABLE_AT broadcast times (38234 events / 868 symbols). Surprise construction can use YoY EPS from PIT fundamentals when present.
- **event_reactions_announcements** — Corporate announcement / result-update events with exchange timestamps.
- **quality_profitability** — PIT fundamentals (EPS/PAT/revenue) with AVAILABLE_AT from NSE XBRL (12584 rows / 860 symbols).
- **earnings_growth** — Sequential / YoY EPS and profit fields are PIT-dated via available_at.
- **value** — Trailing PE derived from PIT EPS × certified close at available_at (10427 rows / 813 symbols).

### PARTIAL


### Still blocked

- **ownership_shareholding_effects** (`DATA_MISSING`) — No PIT shareholding ledger with filing AVAILABLE_AT yet.
- **sector_neutral_fundamental_factors** (`PIT_UNSAFE`) — PIT sector membership history still NOT_RESEARCH_READY.

### Closed (not reopened)

- momentum
- reversal
- low_volatility
- network_alpha
- vol_compression

## 8. What remains NOT research-ready

- PIT sector membership history
- Shareholding / ownership with filing AVAILABLE_AT
- Analyst-consensus earnings surprise (YoY XBRL surprise is an interim proxy)
- Full-universe RESEARCH_GRADE upgrade (global still OPERATIONAL_ONLY)
- Balance-sheet book value / ROE completeness depends on XBRL tag coverage

## 9. Production behaviour confirmation

| Surface | Status |
|---------|--------|
| Brain / ranking / risk / execution | Unchanged |
| Screener fundamentals cache | Unchanged (still operational UI) |
| Live trading | Unchanged |

## 10. What NOT to build next

- ML/AI to invent fundamentals
- Strategies in this foundation PR
- Fabricated AVAILABLE_AT from scrape time
- Reopening closed price-factor branches

## Status card

| Field | Value |
|-------|--------|
| FOUNDATION SNAPSHOT | `46ff79f58ee21c9e` |
| EVENTS | 43569 rows / 868 symbols |
| FUNDAMENTALS | 12584 rows / 860 symbols |
| VALUATIONS | 10427 rows / 813 symbols |
| DATA QUALITY | Scoped READY; global OPERATIONAL_ONLY |
| NEW READY FAMILIES | post_earnings_drift_timing, event_reactions_announcements, quality_profitability, earnings_growth, value |
| NEXT SCIENTIFIC ACTION | Preregister ONE READY family experiment (no ML) |

_Generated 2026-08-11T18:06:02.137980+00:00_
_git_sha `3937245fd4a200ebb42ce08c1c0b33bd9d54d803`_
