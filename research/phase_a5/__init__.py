"""
Phase A.5 — Evidence Activation (research only).

Runs preregistered experiments against Phase A infrastructure.
Exploratory panels are stamped DISPLAY_ONLY / LIMITED_RESEARCH and cannot
support PASS_ALPHA / PASS_RISK promotion without RESEARCH_GRADE retest.
Production behaviour is never modified.
"""
from research.phase_a5.run_all import run_phase_a5

__all__ = ["run_phase_a5"]
