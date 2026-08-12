"""Retail conviction shortlist projected from the canonical market scan.

This module does not scan, trade, or mutate state.  It combines the scanner's
stock-level evidence with the current market/sector context and makes the
remaining risks explicit.  The word "conviction" therefore means multiple
independent confirmations, not certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class ConvictionCandidate:
    symbol: str
    company: str
    classification: str
    conviction_score: float
    scanner_score: float
    price: float
    entry: float
    stop: float
    target: float
    rsi: float
    volume_ratio: float
    market_health: str
    sector: str
    status: str
    reasons: tuple[str, ...]
    risks: tuple[str, ...]

    def as_dict(self) -> dict:
        data = asdict(self)
        data["reasons"] = list(self.reasons)
        data["risks"] = list(self.risks)
        return data


def _f(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def build_conviction_shortlist(
    payload: Mapping[str, Any] | None,
    market_view: Any,
    *,
    sector_lookup: Callable[[str], str] | None = None,
) -> list[dict]:
    """Return deterministic conviction rows from one saved canonical scan."""
    if not payload:
        return []
    if sector_lookup is None:
        try:
            from scan.sector_heat import sector_of
            sector_lookup = sector_of
        except Exception:
            sector_lookup = lambda _symbol: ""

    health = str(getattr(market_view, "health", "Mixed") or "Mixed")
    leaders = {str(x).lower() for x in (getattr(market_view, "leaders", ()) or ())}
    laggards = {str(x).lower() for x in (getattr(market_view, "laggards", ()) or ())}
    rows: list[ConvictionCandidate] = []

    for raw in payload.get("records", []) or []:
        if not isinstance(raw, Mapping):
            continue
        symbol = str(raw.get("symbol", "") or "").upper()
        if not symbol:
            continue
        status = str(raw.get("status", "Watch") or "Watch")
        verdict = str(raw.get("verdict", "WATCH") or "WATCH").upper()
        chase = bool(raw.get("chase_risk"))
        scanner_score = _f(raw.get("score"))
        volume = _f(raw.get("volume_ratio"))
        rsi = _f(raw.get("rsi"))
        sector = str(raw.get("sector") or sector_lookup(symbol) or "Unknown")
        sector_key = sector.lower()

        score = scanner_score * 0.70
        reasons = [str(x) for x in (raw.get("reasons") or []) if str(x).strip()][:3]
        risks: list[str] = []

        if health.lower() == "healthy":
            score += 10
            reasons.append("Market regime supportive")
        elif health.lower() == "weak":
            score -= 15
            risks.append("Market regime weak")

        if sector_key in leaders:
            score += 10
            reasons.append(f"Leading sector: {sector}")
        elif sector_key in laggards:
            score -= 10
            risks.append(f"Lagging sector: {sector}")

        score += max(0.0, min(10.0, (volume - 1.0) * 7.0))
        if volume >= 2.0:
            reasons.append(f"Volume {volume:.1f}× normal")
        elif volume < 1.5:
            risks.append("Volume confirmation is weak")

        if 50 <= rsi <= 70:
            score += 5
        elif rsi >= 82:
            score -= 15
            risks.append(f"RSI {rsi:.0f}: blow-off/chase risk")
        elif rsi >= 72:
            score -= 5
            risks.append(f"RSI {rsi:.0f}: extended")

        if status == "Ready to trade" and verdict in ("BUY", "STRONG BUY"):
            score += 5
        elif status == "Watch for breakout":
            risks.append("Trigger not confirmed yet")
        if chase or status == "Wait for pullback":
            score -= 20
            risks.append("Price is extended; wait for pullback")

        score = round(max(0.0, min(100.0, score)), 1)
        if chase or status == "Wait for pullback":
            classification = "WAIT_FOR_PULLBACK"
        elif (score >= 75 and status == "Ready to trade" and
              health.lower() != "weak" and sector_key not in laggards):
            classification = "HIGH_CONVICTION"
        elif status in ("Ready to trade", "Watch for breakout"):
            classification = "AWAIT_CONFIRMATION"
        else:
            classification = "WATCH"

        rows.append(ConvictionCandidate(
            symbol=symbol,
            company=str(raw.get("company", symbol) or symbol),
            classification=classification,
            conviction_score=score,
            scanner_score=scanner_score,
            price=_f(raw.get("price")), entry=_f(raw.get("entry")),
            stop=_f(raw.get("stop")), target=_f(raw.get("target")),
            rsi=rsi, volume_ratio=volume, market_health=health,
            sector=sector, status=status, reasons=tuple(dict.fromkeys(reasons)),
            risks=tuple(dict.fromkeys(risks)),
        ))

    priority = {"HIGH_CONVICTION": 0, "AWAIT_CONFIRMATION": 1,
                "WAIT_FOR_PULLBACK": 2, "WATCH": 3}
    rows.sort(key=lambda r: (priority.get(r.classification, 9),
                             -r.conviction_score, r.symbol))
    return [row.as_dict() for row in rows]
