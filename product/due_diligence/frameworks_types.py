"""Shared types for sector frameworks. Kept out of frameworks.py to avoid import cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from product.due_diligence.sector_frameworks.kpis import KpiSpec


@dataclass(frozen=True)
class SectorFramework:
    id: str
    label: str
    kpis: tuple[KpiSpec, ...]
    blurb: str = ""
    min_critical: int = 2
    min_decision_coverage: float = 0.45
    min_score_coverage: float = 0.40
    watch: tuple[str, ...] = ()
    acquire_priority: tuple[str, ...] = ("financials", "ir", "filings", "news")
    material_tokens: tuple[str, ...] = ()
    skip_cfo_conversion: bool = False
    cycle_aware: bool = False
    lending: bool = False
    peer_note: str = ""
    default_sub_sector: str = ""
    default_business_model: str = ""

    def spec(self, kpi_id: str) -> KpiSpec | None:
        for item in self.kpis:
            if item.id == kpi_id:
                return item
        return None

    def required_ids(self) -> tuple[str, ...]:
        return tuple(k.id for k in self.kpis if k.importance in ("critical", "important"))

    def critical_ids(self) -> tuple[str, ...]:
        return tuple(k.id for k in self.kpis if k.importance == "critical")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "blurb": self.blurb,
            "kpis": self.kpis,
            "min_critical": self.min_critical,
            "min_decision_coverage": self.min_decision_coverage,
            "min_score_coverage": self.min_score_coverage,
            "watch": self.watch,
            "acquire_priority": self.acquire_priority,
            "material_tokens": self.material_tokens,
            "skip_cfo_conversion": self.skip_cfo_conversion,
            "cycle_aware": self.cycle_aware,
            "lending": self.lending,
            "peer_note": self.peer_note,
            "default_sub_sector": self.default_sub_sector,
            "default_business_model": self.default_business_model,
        }
