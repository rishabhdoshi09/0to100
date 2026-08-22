# FEATURE-001 forward ledger

Documented only. Do not wire into `app.py` in this milestone.

```json
{
  "kind": "FEATURE_001_SHADOW_FEATURE_LOG",
  "experiment": "FEATURE-001",
  "execution": false,
  "paper": false,
  "autopilot": false,
  "activation": "documented_only",
  "reason": "Wiring passive logging into app.py / auto_scan could change production runtime reliability. FEATURE-001 only documents the ledger. Activate in a later milestone behind an explicit flag that never feeds the ticket, GTT, or autopilot.",
  "eligible_after": "strictly after the FEATURE-001 experiment freeze date",
  "fields": [
    "symbol",
    "timestamp",
    "strategy",
    "production_rank",
    "production_score",
    "production_verdict",
    "actual_production_decision",
    "trend_features_v1_vector",
    "rs_features_v1_vector",
    "existing_conviction_technical_only",
    "hypothetical_feature_adjusted_rank",
    "future_outcome_when_known"
  ],
  "forbidden": [
    "change BUY/WATCH",
    "change ranking used by Ready/autopilot",
    "place or cancel orders",
    "feed Telegram as a trade licence"
  ]
}
```
