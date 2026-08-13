# A4 — CycleContext Extension Seams

**Milestone:** Phase A / A4  
**Status:** Optional typed seams only — **no Brain/execution behaviour change**  
**Authority:** `QUANTTERM_INSTITUTIONAL_AI_AUDIT.md`

## Goal

Prove future research outputs (market structure, network risk, horizon views,
challenger evidence) can attach to the existing canonical `CycleContext`
without inventing a second decision-state object.

## Change

- Added optional fields on `CycleContext`, defaulting to `None`:
  - `market_structure`
  - `network_risk`
  - `horizon_view`
  - `challenger_evidence`
- Typed containers in `research/intelligence/runtime/research_seams.py`
- `cycle_id()` identity is **unchanged** (seams are not part of cycle identity)
- No producer populates these fields yet (A5/A6 will)

## Non-goals

- No DecisionResolver
- No Brain/allocation/execution branching on these fields
- No mandatory population
