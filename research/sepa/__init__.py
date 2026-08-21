"""SEPA-001 canonical eligibility (research only)."""
from __future__ import annotations

from research.sepa.config import DEFAULT_CONFIG, ELIGIBILITY_VERSION, SepaConfig
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.types import SepaEligibility

__all__ = [
    "DEFAULT_CONFIG",
    "ELIGIBILITY_VERSION",
    "SepaConfig",
    "SepaEligibility",
    "evaluate_sepa_eligibility",
]
