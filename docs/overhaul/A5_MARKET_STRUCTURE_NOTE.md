# A5 — Market Structure Discovery (Research Only)

**Milestone:** Phase A / A5  
**Status:** Research-only — **zero production decision authority**  
**Authority:** `QUANTTERM_INSTITUTIONAL_AI_AUDIT.md`

## Goal

Infer changing cross-sectional structure from PIT returns using transparent
methods, then measure stability and agreement with existing sector / correlation
/ regime lenses. Do **not** assume alpha.

## Methods (initial)

1. Hierarchical clustering (average linkage on correlation distance)
2. PCA latent factors
3. k-means baseline on the same feature matrix

## Reuse

| Need | Source |
|------|--------|
| PIT bars | A1 `PitContract` / `Snapshot` (caller supplies returns ≤ as_of) |
| Correlation incumbent | `risk.correlation.clusters_from_corr` for agreement stats |
| CycleContext seam | `MarketStructureView` (optional attach; not auto-wired) |
| Provenance | `research.registry.register_hypothesis` when requested |

## Non-goals

- No spectral / DBSCAN / OPTICS / ICA / autoencoders
- No scanner demotion/boost
- No Brain posture change
- No live order path
