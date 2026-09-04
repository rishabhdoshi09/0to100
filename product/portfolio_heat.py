"""Persist and expose total open risk and correlated heat.

A book of individually 1% trades is still too hot when they are correlated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "logs" / "product" / "portfolio_heat.json"


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def measure(
    positions: Sequence[Mapping[str, Any]],
    *,
    capital: float,
    corr_pairs: Sequence[Mapping[str, Any]] | None = None,
    name_cap_pct: float = 0.01,
) -> dict[str, Any]:
    opens = [p for p in positions if p]
    risks = []
    for pos in opens:
        risk = _f(pos.get("risk_pct") or pos.get("actual_risk_pct"))
        if risk is None:
            entry = _f(pos.get("entry") or pos.get("entry_price"))
            stop = _f(pos.get("stop") or pos.get("stop_price"))
            qty = _f(pos.get("qty") or pos.get("quantity") or pos.get("shares"))
            if entry and stop and qty and capital > 0 and entry > stop:
                risk = ((entry - stop) * qty) / capital * 100.0
        if risk is not None:
            risks.append(risk)
    gross = round(sum(risks), 4)
    n = len(opens)
    corr = list(corr_pairs or [])
    cluster_mult = 1.0
    if corr:
        # Conservative: highly correlated names count closer to one risk unit.
        cluster_mult = min(1.0 + 0.5 * len(corr), 3.0)
    effective = round(gross * (1.0 if not corr else min(cluster_mult, n or 1)), 4)
    return {
        "n_open": n,
        "capital": capital,
        "gross_open_risk_pct": gross,
        "name_cap_pct": name_cap_pct * 100.0,
        "correlated_pairs": corr,
        "effective_correlated_heat_pct": effective,
        "heat_unacceptable": effective >= 5.0,
        "note": (
            "Gross open risk is the sum of frozen per-name rupee risk / capital. "
            "Effective heat inflates when official return correlations cluster."
        ),
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }


def persist(payload: Mapping[str, Any], *, path: Path | None = None) -> Path:
    target = path or PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    return target
