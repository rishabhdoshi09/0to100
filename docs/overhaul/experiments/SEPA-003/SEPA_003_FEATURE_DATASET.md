# SEPA-003 feature dataset

One row per reconstructed unique F fill (frozen `sepa-001r2.v1` /
`vcp_causal_v2` / `pivot_last_contraction_v1`).

| Item | Count |
|---|---|
| Ledger F setups | 5,797 |
| Reconstructed fills with net R | 3,432 |
| CA-censored rows | 2 |
| Stage-2+RS no-VCP controls | 7,932 |
| G panel (Stage-2+RS forward %) | 20,813 |

R2.1 official F n=4,208 is the expectancy record. This table is the
**explanatory** book. Difference: 20-session fill search, no second
embargo pass, MISSED/GAP_THROUGH refusals not forced into R.

## Files

- `sepa_003_features.parquet` (committed)
- `sepa_003_stats.json` (committed)
- `sepa_003_controls.jsonl` / `sepa_003_g_panel.jsonl` (local rebuild;
  36MB combined, regenerable via `python -m research.sepa003`)
- `sepa_003_hypotheses.json` (predeclared)
- `sepa_003_feature_manifest.json`
- `sepa_003_dataset_meta.json`

Every feature row carries `not_validated_edge=true` and
`confirmation_already_observed=true`.
