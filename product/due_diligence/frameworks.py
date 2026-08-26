"""Sector KPI frameworks.

Canonical definitions live in `sector_frameworks/`. This module is the
compatibility shim the existing StockResearchEngine imports.
"""
from __future__ import annotations

from typing import Any

from product.due_diligence.sector_frameworks import (
    FRAMEWORKS as FRAMEWORK_OBJECTS,
    get_framework,
    list_frameworks,
)
from product.due_diligence.sector_frameworks.kpis import (
    AUTO,
    BANK,
    FMCG,
    GENERIC,
    INDUSTRIALS,
    IT,
    KpiSpec,
    METALS,
    NBFC,
    PHARMA,
    REALTY,
    _k,
)

FRAMEWORKS: dict[str, dict[str, Any]] = {
    key: value.as_dict() for key, value in FRAMEWORK_OBJECTS.items()
}

__all__ = [
    "AUTO",
    "BANK",
    "FMCG",
    "FRAMEWORKS",
    "GENERIC",
    "INDUSTRIALS",
    "IT",
    "KpiSpec",
    "METALS",
    "NBFC",
    "PHARMA",
    "REALTY",
    "_k",
    "get_framework",
    "list_frameworks",
]
