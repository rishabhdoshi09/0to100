"""Immutable SEPA research-program status.

SEPA-001 → SEPA-001R → SEPA-001R2/R2.1 → SEPA-003 are a completed
benchmark. Core F is not a production strategy candidate. Surviving
concepts (Trend Template structure, rs_cs_v1) are NEW_HYPOTHESIS
only — FEATURE-001 tests them as features, not as Core SEPA.
"""

from __future__ import annotations

# Do not flip this without a new experiment id and a protocol that
# does not consume the already-used 2019-04 → 2026-03 confirmation
# window as a validation claim.
CORE_SEPA_STATUS = "RETIRED_RESEARCH_BENCHMARK"

SURVIVING_CONCEPTS = (
    "trend_template_structure",
    "rs_cs_v1",
)

SURVIVING_CONCEPT_STATUS = "NEW_HYPOTHESIS"
