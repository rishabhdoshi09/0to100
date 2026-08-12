"""Plain-language forward-evidence status for the retail UI."""
from __future__ import annotations

from research.forward_evidence.service import dashboard_payload, ensure_armed, system_status


def read_forward_evidence_dashboard() -> dict:
    ensure_armed()
    return dashboard_payload()


def read_forward_evidence_status() -> dict:
    return system_status()
