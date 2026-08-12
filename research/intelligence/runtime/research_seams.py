"""
Optional research-derived overlays for CycleContext (Phase A / A4).

These are typed seams only. Producers (market structure / network / horizons /
challenger lab) may attach results later. Brain and execution must not require
them — absence means "not computed", never an implicit signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class MarketStructureView:
    """Research-only market-structure snapshot as-of a date."""

    as_of: str
    method: str = ""
    cluster_id_by_symbol: Mapping[str, int] = field(default_factory=dict)
    cluster_stability: float | None = None
    structural_shift_score: float | None = None
    outlier_score_by_symbol: Mapping[str, float] = field(default_factory=dict)
    latent_factor_exposure: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    meta: Mapping[str, Any] = field(default_factory=dict)
    evidence_status: str = "RESEARCH_ONLY"


@dataclass(frozen=True)
class NetworkRiskView:
    """Research-only portfolio network metrics (advisory until validated)."""

    as_of: str
    community_id_by_symbol: Mapping[str, int] = field(default_factory=dict)
    centrality_by_symbol: Mapping[str, float] = field(default_factory=dict)
    portfolio_network_concentration: float | None = None
    incremental_cluster_risk: Mapping[str, float] = field(default_factory=dict)
    contagion_score: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    evidence_status: str = "RESEARCH_ONLY"


@dataclass(frozen=True)
class HorizonView:
    """Multi-horizon research summary for the cycle date."""

    as_of: str
    per_horizon: Mapping[str, Any] = field(default_factory=dict)
    consensus: Mapping[str, Any] = field(default_factory=dict)
    best_supported_horizon: str | None = None
    horizon_dispersion: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    evidence_status: str = "RESEARCH_ONLY"


@dataclass(frozen=True)
class ChallengerEvidenceView:
    """Latest challenger bake-off evidence attached to a cycle (advisory)."""

    as_of: str
    role: str = ""
    incumbent_id: str = ""
    challenger_id: str = ""
    verdict: str = ""
    hypothesis_id: str = ""
    economic_value_delta: float | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)
    evidence_status: str = "RESEARCH_ONLY"
