# SEPA-003 forward observation (design only)

Not activated. Not paper. Not autopilot.

```json
{
  "kind": "SEPA_003_FORWARD_OBSERVATION",
  "execution": false,
  "paper": false,
  "autopilot": false,
  "fields": [
    "as_of",
    "symbol",
    "frozen_feature_vector",
    "hypothetical_entry",
    "hypothetical_stop",
    "regime_pit_v1",
    "sector_map_v1",
    "data_quality",
    "outcome_after_horizon"
  ],
  "activation": "documented_only",
  "reason": "Wiring this into app.py would change production runtime. SEPA-003 only documents the ledger."
}
```
