# FEATURE-001 — Dataset (regenerable)

The filled-event panel is **too large for git** (~200MB events, ~250MB family rows).

Regenerate locally (official bhavcopy required):

```
python -m research.feature001
```

This writes:

- `feature_001_events.jsonl` — one row per filled `UnifiedScanner._analyze` fire
- `feature_001_family_rows.jsonl` — exploded by `SIGNAL_META` key
- `feature_001_dataset_meta.json` — coverage and sampling
- `feature_001_stats.json` — all study tables (committed)
- `feature_001_per_strategy.json` — family baselines + policy (committed)
- `feature_001_ranking.json` — within-day rank spreads (committed)
- `feature_001_feature_manifest.json` — versions and final status (committed)

Do not treat a regenerated panel as new confirmation data. The 2020-09-28 → 2026-07-23 window is already consumed.
