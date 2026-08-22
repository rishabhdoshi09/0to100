"""SEPA-001 canonical eligibility (research only)."""
from __future__ import annotations

from research.sepa.config import DEFAULT_CONFIG, ELIGIBILITY_VERSION, SepaConfig
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.status import CORE_SEPA_STATUS, SURVIVING_CONCEPTS, SURVIVING_CONCEPT_STATUS
from research.sepa.types import SepaEligibility

__all__ = [
    "CORE_SEPA_STATUS",
    "DEFAULT_CONFIG",
    "ELIGIBILITY_VERSION",
    "SURVIVING_CONCEPTS",
    "SURVIVING_CONCEPT_STATUS",
    "SepaConfig",
    "SepaEligibility",
    "evaluate_sepa_eligibility",
]
