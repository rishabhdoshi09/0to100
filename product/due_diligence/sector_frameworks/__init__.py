"""Central sector/business-model framework registry.

One Investigate engine loads one of these frameworks. Do not add per-sector engines.
"""

from __future__ import annotations

from typing import Any

from product.due_diligence.frameworks_types import SectorFramework

from .catalog import FRAMEWORKS
from .kpis import KpiSpec

__all__ = [
    "FRAMEWORKS",
    "KpiSpec",
    "SectorFramework",
    "get_framework",
    "get_framework_object",
    "list_frameworks",
]


def get_framework_object(framework_id: str) -> SectorFramework:
    return FRAMEWORKS.get(str(framework_id or "").strip()) or FRAMEWORKS["generic"]


def get_framework(framework_id: str) -> dict[str, Any]:
    """Dict form used by the existing StockResearchEngine."""
    return get_framework_object(framework_id).as_dict()


def list_frameworks() -> tuple[str, ...]:
    return tuple(FRAMEWORKS.keys())
