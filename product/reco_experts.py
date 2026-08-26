"""Independent recommendation experts over persisted QuantTerm evidence.

Each expert answers a different question. Missing inputs stay unknown.
No new scanner. No OHLCV rescore on page-open. No invented consensus.

Experts propose; they do not average into a score soup. Cross-sectional
ranks are computed inside the supplied universe (usually the saved scan).
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.breakout_quality import RSI_HARD, has_usable_fundamentals
from product.radar_workspace import is_sniper_breakout_candidate

# Evidence families — correlated price signals collapse into one family.
FAMILY_PRICE = "price_leadership"
FAMILY_STRUCTURE = "structure"
FAMILY_FUND_CHANGE = "fundamental_change"
FAMILY_QUALITY = "business_quality"
FAMILY_PARTICIPATION = "participation"
FAMILY_CATALYST = "catalyst"
FAMILY_CONTEXT = "market_context"

FAMILY_LABELS: dict[str, str] = {
    FAMILY_PRICE: "Price Leadership",
    FAMILY_STRUCTURE: "Structure",
    FAMILY_FUND_CHANGE: "Fundamental Change",
    FAMILY_QUALITY: "Business Quality",
    FAMILY_PARTICIPATION: "Participation",
    FAMILY_CATALYST: "Catalyst",
    FAMILY_CONTEXT: "Market Context",
}

HORIZON_TACTICAL = "tactical"
HORIZON_SWING = "swing"
HORIZON_POSITION = "position"

HORIZON_LABELS: dict[str, str] = {
    HORIZON_TACTICAL: "few sessions",
    HORIZON_SWING: "days to weeks",
    HORIZON_POSITION: "several weeks/months",
}

_BREAKOUT_SIGS = frozenset({"BREAKOUT_52W", "BREAKOUT_RES", "PRE_BREAKOUT"})
_VCP_SIGS = frozenset({"VCP", "NR7_COIL", "FLAT_BASE", "VOL_SQUEEZE", "CUP_HANDLE"})
_DELIVERY_SIGS = frozenset({"DELIVERY_SPIKE", "ACCUMULATION", "POCKET_PIVOT"})
_QUALITY_CLASSES = frozenset({
    "QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE",
})
_ACCEL_TOKENS = ("accelerat", "inflection", "sequential", "qoq", "q/q")


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _signals(row: Mapping[str, Any]) -> set[str]:
    return {str(x).upper() for x in (row.get("signals") or [])}


def _nested_fund(row: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = row.get("fundamentals")
    return raw if isinstance(raw, Mapping) else {}


def _expert(
    *,
    id_: str,
    label: str,
    family: str,
    status: str,
    eligible: bool,
    score: float | None,
    thesis: str,
    horizon: str,
    evidence: Sequence[str],
    rank: int | None = None,
    quality_flags: Sequence[str] | None = None,
    freshness: str = "saved_scan",
) -> dict[str, Any]:
    return {
        "id": id_,
        "label": label,
        "family": family,
        "family_label": FAMILY_LABELS.get(family, family),
        "status": status,
        "eligible": bool(eligible),
        "score": None if score is None else round(float(score), 1),
        "rank": rank,
        "thesis": thesis,
        "horizon": horizon,
        "horizon_label": HORIZON_LABELS.get(horizon, horizon),
        "evidence": [str(x) for x in evidence if x],
        "freshness": freshness,
        "quality_flags": [str(x) for x in (quality_flags or []) if x],
    }


def _percentile(rank: int, n: int) -> float:
    if n <= 0:
        return 0.0
    return 100.0 * (n - rank + 1) / n


def _rank_map(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, tuple[int, int, float]]:
    """symbol → (rank 1=best, n, value) for a numeric field. Missing excluded."""
    scored: list[tuple[str, float]] = []
    for row in rows:
        sym = str(row.get("symbol") or "").upper()
        val = _f(row.get(key))
        if not sym or val is None:
            continue
        scored.append((sym, val))
    scored.sort(key=lambda item: (-item[1], item[0]))
    n = len(scored)
    out: dict[str, tuple[int, int, float]] = {}
    for i, (sym, val) in enumerate(scored, start=1):
        out[sym] = (i, n, val)
    return out


def _xs_momentum_expert(row: Mapping[str, Any], ranks: Mapping[str, tuple[int, int, float]]) -> dict[str, Any]:
    """Cross-sectional momentum from persisted 5d (and 6m/12m only if already on the row)."""
    sym = str(row.get("symbol") or "").upper()
    mom5 = _f(row.get("momentum_5d"))
    mom6 = _f(row.get("momentum_6m") or row.get("mom_6m") or row.get("return_6m"))
    mom12 = _f(row.get("momentum_12m") or row.get("mom_12m") or row.get("return_12m"))
    vol_adj = _f(row.get("vol_adj_momentum") or row.get("momentum_vol_adj"))
    hit = ranks.get(sym)
    evidence: list[str] = []
    flags: list[str] = []
    if mom6 is None and mom12 is None:
        flags.append("6m/12m momentum not on file — not invented")
    if mom5 is None and mom6 is None and mom12 is None:
        return _expert(
            id_="xs_momentum", label="Cross-sectional momentum", family=FAMILY_PRICE,
            status="unknown", eligible=False, score=None,
            thesis="Cross-sectional momentum", horizon=HORIZON_SWING,
            evidence=["No persisted momentum print on file"],
            quality_flags=flags,
        )
    if mom5 is not None:
        evidence.append(f"5-day momentum {mom5:+.1f}%")
    if mom6 is not None:
        evidence.append(f"6-month momentum {mom6:+.1f}%")
    if mom12 is not None:
        evidence.append(f"12-month momentum {mom12:+.1f}%")
    if vol_adj is not None:
        evidence.append(f"Vol-adjusted momentum {vol_adj:.2f}")
    # Negative 5d is fading leadership, not a momentum proposal.
    if mom5 is not None and mom5 < 0 and (mom6 is None or mom6 < 0):
        return _expert(
            id_="xs_momentum", label="Cross-sectional momentum", family=FAMILY_PRICE,
            status="fail", eligible=False, score=max(0.0, 50.0 + (mom5 or 0)),
            thesis="Cross-sectional momentum", horizon=HORIZON_SWING,
            evidence=evidence, rank=hit[0] if hit else None, quality_flags=flags,
        )
    pctl = _percentile(hit[0], hit[1]) if hit else None
    rank = hit[0] if hit else None
    n = hit[1] if hit else 0
    if pctl is not None:
        evidence.append(f"Cross-section rank {rank}/{n} (pctl {pctl:.0f})")
    sigs = _signals(row)
    strong_print = (
        (pctl is not None and pctl >= 80.0 and (mom5 or 0) > 0)
        or ("MOMENTUM" in sigs and pctl is not None and pctl >= 60.0 and (mom5 or 0) > 0)
        or (mom6 is not None and mom6 >= 20 and (mom12 is None or mom12 > 0))
    )
    if strong_print:
        score = min(100.0, (pctl or 70.0) + (8.0 if "MOMENTUM" in sigs else 0.0))
        return _expert(
            id_="xs_momentum", label="Cross-sectional momentum", family=FAMILY_PRICE,
            status="pass", eligible=True, score=score,
            thesis="Cross-sectional momentum", horizon=HORIZON_SWING,
            evidence=evidence, rank=rank, quality_flags=flags,
        )
    if pctl is not None and pctl < 40.0:
        return _expert(
            id_="xs_momentum", label="Cross-sectional momentum", family=FAMILY_PRICE,
            status="fail", eligible=False, score=pctl,
            thesis="Cross-sectional momentum", horizon=HORIZON_SWING,
            evidence=evidence, rank=rank, quality_flags=flags,
        )
    return _expert(
        id_="xs_momentum", label="Cross-sectional momentum", family=FAMILY_PRICE,
        status="neutral", eligible=False, score=pctl,
        thesis="Cross-sectional momentum", horizon=HORIZON_SWING,
        evidence=evidence or ["Momentum mid-pack — not a proposal"],
        rank=rank, quality_flags=flags,
    )


def _sepa_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    score = _f(row.get("sepa_score"))
    if score is None:
        return _expert(
            id_="sepa", label="SEPA / growth leadership", family=FAMILY_PRICE,
            status="unknown", eligible=False, score=None,
            thesis="SEPA leader", horizon=HORIZON_POSITION,
            evidence=["No persisted SEPA overlay for this name"],
        )
    bits = [f"SEPA overlay {score:.0f}/100"]
    for key, label in (
        ("sepa_trend", "trend template"),
        ("sepa_rs", "relative strength"),
        ("sepa_base", "constructive base"),
    ):
        val = row.get(key)
        if val not in (None, "", False):
            bits.append(label)
    if score >= 40:
        return _expert(
            id_="sepa", label="SEPA / growth leadership", family=FAMILY_PRICE,
            status="pass", eligible=True, score=min(100.0, score),
            thesis="SEPA leader", horizon=HORIZON_POSITION, evidence=bits,
        )
    return _expert(
        id_="sepa", label="SEPA / growth leadership", family=FAMILY_PRICE,
        status="fail", eligible=False, score=score,
        thesis="SEPA leader", horizon=HORIZON_POSITION, evidence=bits,
    )


def _rs_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    pctl = _f(row.get("rs_percentile"))
    vs = _f(row.get("rs_vs_nifty") or row.get("rs_vs_nifty_20d") or row.get("rs_score"))
    if pctl is None and vs is None:
        return _expert(
            id_="rs", label="Relative strength", family=FAMILY_PRICE,
            status="unknown", eligible=False, score=None,
            thesis="RS leadership", horizon=HORIZON_SWING,
            evidence=["No RS percentile or vs-Nifty print on file"],
        )
    if pctl is not None:
        evidence = [f"RS percentile {pctl:.0f}"]
        if pctl >= 70:
            return _expert(
                id_="rs", label="Relative strength", family=FAMILY_PRICE,
                status="pass", eligible=True, score=pctl,
                thesis="RS leadership", horizon=HORIZON_SWING, evidence=evidence,
            )
        if pctl < 40:
            return _expert(
                id_="rs", label="Relative strength", family=FAMILY_PRICE,
                status="fail", eligible=False, score=pctl,
                thesis="RS leadership", horizon=HORIZON_SWING, evidence=evidence,
            )
        return _expert(
            id_="rs", label="Relative strength", family=FAMILY_PRICE,
            status="neutral", eligible=False, score=pctl,
            thesis="RS leadership", horizon=HORIZON_SWING,
            evidence=[f"RS percentile {pctl:.0f} is mid-pack"],
        )
    assert vs is not None
    evidence = [f"RS vs Nifty {vs:+.1f}"]
    if vs > 0:
        return _expert(
            id_="rs", label="Relative strength", family=FAMILY_PRICE,
            status="pass", eligible=True, score=min(100.0, 55.0 + vs),
            thesis="RS leadership", horizon=HORIZON_SWING, evidence=evidence,
        )
    if vs < 0:
        return _expert(
            id_="rs", label="Relative strength", family=FAMILY_PRICE,
            status="fail", eligible=False, score=max(0.0, 50.0 + vs),
            thesis="RS leadership", horizon=HORIZON_SWING, evidence=evidence,
        )
    return _expert(
        id_="rs", label="Relative strength", family=FAMILY_PRICE,
        status="neutral", eligible=False, score=50.0,
        thesis="RS leadership", horizon=HORIZON_SWING, evidence=evidence,
    )


def _breakout_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    if bool(row.get("chase_risk")):
        return _expert(
            id_="breakout", label="Breakout quality", family=FAMILY_STRUCTURE,
            status="fail", eligible=False, score=0.0,
            thesis="Breakout", horizon=HORIZON_TACTICAL,
            evidence=["Chase / extension — breakout expert rejects"],
        )
    sigs = _signals(row)
    grade = str(row.get("breakout_grade") or "").upper()
    sniper = is_sniper_breakout_candidate(row)
    visible = bool(sigs & _BREAKOUT_SIGS) or str(row.get("status") or "") == "Watch for breakout"
    if not sniper and grade not in {"A", "B"} and not visible:
        return _expert(
            id_="breakout", label="Breakout quality", family=FAMILY_STRUCTURE,
            status="unknown", eligible=False, score=None,
            thesis="Breakout", horizon=HORIZON_TACTICAL,
            evidence=["No breakout tag, grade, or sniper confirm on file"],
        )
    bits = []
    if grade:
        bits.append(f"Grade {grade}")
    hits = sorted(sigs & _BREAKOUT_SIGS)
    if hits:
        bits.append("+".join(hits))
    dist = _f(row.get("pivot_distance_pct"))
    if dist is not None:
        bits.append(f"Pivot distance {dist:.1f}%")
    vol = _f(row.get("volume_ratio"))
    if vol is not None and vol > 0:
        bits.append(f"Volume {vol:.2f}×")
    else:
        bits.append("Volume not on file")
    if sniper or grade in {"A", "B"}:
        pts = 90.0 if sniper else 80.0
        if grade == "A":
            pts = min(100.0, pts + 8)
        return _expert(
            id_="breakout", label="Breakout quality", family=FAMILY_STRUCTURE,
            status="pass", eligible=True, score=pts,
            thesis="Breakout", horizon=HORIZON_TACTICAL, evidence=bits,
        )
    return _expert(
        id_="breakout", label="Breakout quality", family=FAMILY_STRUCTURE,
        status="neutral", eligible=False, score=55.0,
        thesis="Breakout", horizon=HORIZON_TACTICAL,
        evidence=bits + ["Ungraded scan breakout — watch, not a proposal"],
    )


def _vcp_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    sigs = _signals(row)
    hits = sorted(sigs & _VCP_SIGS)
    if not hits:
        return _expert(
            id_="vcp", label="VCP / compression", family=FAMILY_STRUCTURE,
            status="unknown", eligible=False, score=None,
            thesis="VCP + sector leadership", horizon=HORIZON_SWING,
            evidence=["No VCP / coil / squeeze tag on file"],
        )
    bits = ["+".join(hits)]
    if bool(row.get("chase_risk")):
        return _expert(
            id_="vcp", label="VCP / compression", family=FAMILY_STRUCTURE,
            status="fail", eligible=False, score=0.0,
            thesis="VCP + sector leadership", horizon=HORIZON_SWING,
            evidence=bits + ["Extended — compression is not an entry"],
        )
    return _expert(
        id_="vcp", label="VCP / compression", family=FAMILY_STRUCTURE,
        status="pass", eligible=True, score=78.0 if "VCP" in hits else 68.0,
        thesis="VCP + sector leadership", horizon=HORIZON_SWING, evidence=bits,
    )


def _pullback_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    """Strong trend, constructive pause — not a Ready-to-trade continuation."""
    status = str(row.get("status") or "")
    rsi = _f(row.get("rsi"))
    above200 = row.get("above_sma200")
    above50 = row.get("above_sma50")
    wait = "pullback" in status.lower() or status == "Wait for pullback"
    chase = bool(row.get("chase_risk"))
    grade = str(row.get("breakout_grade") or "").upper()
    ready = status == "Ready to trade" or grade in {"A", "B"}
    trend_ok = above50 is True or above200 is True
    cooling = rsi is not None and 40.0 <= rsi <= 62.0
    if rsi is not None and rsi > RSI_HARD:
        return _expert(
            id_="pullback", label="Trend pullback", family=FAMILY_STRUCTURE,
            status="fail", eligible=False, score=0.0,
            thesis="Trend pullback", horizon=HORIZON_SWING,
            evidence=[f"RSI {rsi:.0f} is blow-off, not a constructive pause"],
        )
    if ready and not wait:
        return _expert(
            id_="pullback", label="Trend pullback", family=FAMILY_STRUCTURE,
            status="unknown", eligible=False, score=None,
            thesis="Trend pullback", horizon=HORIZON_SWING,
            evidence=["Ready/graded tape is not a pullback setup"],
        )
    if chase and not trend_ok and not wait:
        return _expert(
            id_="pullback", label="Trend pullback", family=FAMILY_STRUCTURE,
            status="fail", eligible=False, score=0.0,
            thesis="Trend pullback", horizon=HORIZON_SWING,
            evidence=["Chase without readable trend structure"],
        )
    constructive = wait or (trend_ok and cooling and not ready)
    if constructive and (trend_ok or wait):
        bits = []
        if wait:
            bits.append(status or "Wait for pullback")
        if above200 is True:
            bits.append("Above SMA200")
        elif above50 is True:
            bits.append("Above SMA50")
        if cooling:
            bits.append(f"RSI {rsi:.0f} cooling into trend")
        if chase:
            bits.append("Still extended — pullback is the thesis, not a chase buy")
        return _expert(
            id_="pullback", label="Trend pullback", family=FAMILY_STRUCTURE,
            status="pass", eligible=not chase,
            score=72.0 if not chase else 58.0,
            thesis="Trend pullback", horizon=HORIZON_SWING, evidence=bits,
        )
    return _expert(
        id_="pullback", label="Trend pullback", family=FAMILY_STRUCTURE,
        status="unknown", eligible=False, score=None,
        thesis="Trend pullback", horizon=HORIZON_SWING,
        evidence=["No constructive pullback on file"],
    )


def _quality_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    cls = str(row.get("classification") or "")
    if cls == "AVOID_REVIEW":
        return _expert(
            id_="quality", label="Business quality", family=FAMILY_QUALITY,
            status="fail", eligible=False, score=0.0,
            thesis="Momentum + Quality", horizon=HORIZON_POSITION,
            evidence=["AVOID_REVIEW — fundamentals reject"],
        )
    cov = _f(row.get("fundamental_coverage"))
    if has_usable_fundamentals(row):
        fund = _f(row.get("fundamental_score")) or 70.0
        bits = [cls.replace("_", " ") or "Quality class", f"coverage {(cov or 0) * 100:.0f}%"]
        for factor in list(row.get("quality_factors") or [])[:2]:
            bits.append(str(factor))
        return _expert(
            id_="quality", label="Business quality", family=FAMILY_QUALITY,
            status="pass", eligible=True, score=min(100.0, fund),
            thesis="Momentum + Quality", horizon=HORIZON_POSITION, evidence=bits,
        )
    if cov is not None and 0 < cov < 0.50 and cls:
        return _expert(
            id_="quality", label="Business quality", family=FAMILY_QUALITY,
            status="fail", eligible=False, score=0.0,
            thesis="Momentum + Quality", horizon=HORIZON_POSITION,
            evidence=[f"Coverage {cov:.0%} < 50%"],
        )
    if not cls and cov is None and row.get("fundamental_score") is None:
        return _expert(
            id_="quality", label="Business quality", family=FAMILY_QUALITY,
            status="unknown", eligible=False, score=None,
            thesis="Momentum + Quality", horizon=HORIZON_POSITION,
            evidence=["No long-term fundamental row on file"],
        )
    return _expert(
        id_="quality", label="Business quality", family=FAMILY_QUALITY,
        status="unknown", eligible=False, score=None,
        thesis="Momentum + Quality", horizon=HORIZON_POSITION,
        evidence=["Fundamentals present but not a quality class with ≥50% coverage"],
    )


def _earnings_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    """Earnings momentum — sequential acceleration only. 3y CAGR is not acceleration."""
    fund = _nested_fund(row)
    qoq_sales = _f(row.get("sales_growth_qoq") or fund.get("sales_growth_qoq"))
    qoq_pat = _f(row.get("pat_growth_qoq") or row.get("eps_growth_qoq") or fund.get("pat_growth_qoq"))
    surprise = _f(row.get("earnings_surprise") or fund.get("earnings_surprise"))
    sales3 = _f(row.get("sales_growth_3y") or fund.get("sales_growth_3y"))
    profit3 = _f(row.get("profit_growth_3y") or fund.get("profit_growth_3y"))
    factors = " ".join(str(x).lower() for x in (row.get("quality_factors") or []))
    sequential = qoq_sales is not None or qoq_pat is not None
    flags = ["No consensus estimates — observable filings only"]
    if surprise is not None:
        flags.append("Surprise print on file (not fabricated)")
    if sequential:
        bits = []
        if qoq_sales is not None:
            bits.append(f"QoQ sales {qoq_sales:+.1f}%")
        if qoq_pat is not None:
            bits.append(f"QoQ profit {qoq_pat:+.1f}%")
        accel = (qoq_pat is not None and qoq_pat >= 8) or (qoq_sales is not None and qoq_sales >= 8)
        if accel:
            return _expert(
                id_="earnings", label="Earnings momentum", family=FAMILY_FUND_CHANGE,
                status="pass", eligible=True, score=min(100.0, 60.0 + (qoq_pat or qoq_sales or 0)),
                thesis="Earnings inflection + breakout", horizon=HORIZON_SWING,
                evidence=bits, quality_flags=flags,
            )
        return _expert(
            id_="earnings", label="Earnings momentum", family=FAMILY_FUND_CHANGE,
            status="neutral", eligible=False, score=50.0,
            thesis="Earnings inflection + breakout", horizon=HORIZON_SWING,
            evidence=bits or ["Sequential prints present but not accelerating"],
            quality_flags=flags,
        )
    if any(tok in factors for tok in _ACCEL_TOKENS):
        hits = [str(x) for x in (row.get("quality_factors") or []) if any(t in str(x).lower() for t in _ACCEL_TOKENS)]
        return _expert(
            id_="earnings", label="Earnings momentum", family=FAMILY_FUND_CHANGE,
            status="pass", eligible=True, score=70.0,
            thesis="Earnings inflection + breakout", horizon=HORIZON_SWING,
            evidence=hits[:3], quality_flags=flags,
        )
    if sales3 is not None or profit3 is not None:
        bits = []
        if sales3 is not None:
            bits.append(f"3y sales CAGR {sales3:.1f}%")
        if profit3 is not None:
            bits.append(f"3y profit CAGR {profit3:.1f}%")
        bits.append("Multi-year CAGR is growth, not sequential acceleration")
        return _expert(
            id_="earnings", label="Earnings momentum", family=FAMILY_FUND_CHANGE,
            status="neutral", eligible=False, score=None,
            thesis="Earnings inflection + breakout", horizon=HORIZON_SWING,
            evidence=bits, quality_flags=flags + ["Sequential acceleration not measured"],
        )
    return _expert(
        id_="earnings", label="Earnings momentum", family=FAMILY_FUND_CHANGE,
        status="unknown", eligible=False, score=None,
        thesis="Earnings inflection + breakout", horizon=HORIZON_SWING,
        evidence=["No sequential earnings/sales acceleration on file"],
        quality_flags=flags,
    )


def _sector_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    if bool(row.get("sector_laggard")):
        return _expert(
            id_="sector", label="Sector leadership", family=FAMILY_CONTEXT,
            status="fail", eligible=False, score=0.0,
            thesis="Strong stock + weak sector", horizon=HORIZON_SWING,
            evidence=[f"Lagging sector: {row.get('sector') or '—'}"],
            quality_flags=["Conflict exposed — not auto-rejected"],
        )
    if bool(row.get("sector_leader")):
        return _expert(
            id_="sector", label="Sector leadership", family=FAMILY_CONTEXT,
            status="pass", eligible=True, score=80.0,
            thesis="Strong stock + strong sector", horizon=HORIZON_SWING,
            evidence=[f"Leading sector: {row.get('sector') or '—'}"],
        )
    return _expert(
        id_="sector", label="Sector leadership", family=FAMILY_CONTEXT,
        status="unknown", eligible=False, score=None,
        thesis="Sector leadership", horizon=HORIZON_SWING,
        evidence=["Sector pack not in current leaders/laggards"],
    )


def _volume_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    """Independent participation — not the same volume bar that graded the breakout."""
    sigs = _signals(row)
    hits = sorted(sigs & _DELIVERY_SIGS)
    if hits:
        return _expert(
            id_="volume", label="Participation", family=FAMILY_PARTICIPATION,
            status="pass", eligible=True, score=76.0,
            thesis="Participation", horizon=HORIZON_TACTICAL,
            evidence=["+".join(hits)],
        )
    vol = _f(row.get("volume_ratio"))
    if vol is None or vol <= 0:
        return _expert(
            id_="volume", label="Participation", family=FAMILY_PARTICIPATION,
            status="unknown", eligible=False, score=None,
            thesis="Participation", horizon=HORIZON_TACTICAL,
            evidence=["Volume not on file — not treated as a fail"],
        )
    return _expert(
        id_="volume", label="Participation", family=FAMILY_PARTICIPATION,
        status="unknown", eligible=False, score=None,
        thesis="Participation", horizon=HORIZON_TACTICAL,
        evidence=[
            f"Volume {vol:.2f}× is already in the tape/breakout reading — not a second family",
        ],
        quality_flags=["One large-volume candle is not institutional buying"],
    )


def _catalyst_expert(row: Mapping[str, Any]) -> dict[str, Any]:
    events = row.get("material_events") or row.get("catalysts") or []
    if isinstance(events, (list, tuple)) and events:
        headlines = []
        for item in events[:3]:
            if isinstance(item, Mapping):
                headlines.append(str(item.get("headline") or item.get("event_type") or ""))
            else:
                headlines.append(str(item))
        headlines = [h for h in headlines if h]
        if headlines:
            return _expert(
                id_="catalyst", label="Event catalyst", family=FAMILY_CATALYST,
                status="pass", eligible=True, score=70.0,
                thesis="Post-earnings continuation", horizon=HORIZON_TACTICAL,
                evidence=headlines,
            )
    return _expert(
        id_="catalyst", label="Event catalyst", family=FAMILY_CATALYST,
        status="unknown", eligible=False, score=None,
        thesis="Post-earnings continuation", horizon=HORIZON_TACTICAL,
        evidence=["No confirmed material event on this row"],
        quality_flags=["Headline sentiment is not used"],
    )


def _mom_quality_expert(momentum: Mapping[str, Any], quality: Mapping[str, Any]) -> dict[str, Any]:
    """Thesis generator: strong price leadership AND business quality. Not a third family."""
    mom_ok = momentum.get("status") == "pass"
    qual_ok = quality.get("status") == "pass"
    if mom_ok and qual_ok:
        score = min(100.0, 0.55 * float(momentum.get("score") or 70) + 0.45 * float(quality.get("score") or 70))
        return _expert(
            id_="mom_quality", label="Momentum + Quality", family=FAMILY_PRICE,
            status="pass", eligible=True, score=score,
            thesis="Momentum + Quality", horizon=HORIZON_POSITION,
            evidence=(list(momentum.get("evidence") or [])[:2] + list(quality.get("evidence") or [])[:2]),
            quality_flags=["Thesis expert — does not add a third evidence family"],
        )
    if quality.get("status") == "fail" and mom_ok:
        return _expert(
            id_="mom_quality", label="Momentum + Quality", family=FAMILY_PRICE,
            status="fail", eligible=False, score=0.0,
            thesis="Momentum + Quality", horizon=HORIZON_POSITION,
            evidence=["Momentum present but quality rejects"],
            quality_flags=["Thesis expert — does not add a third evidence family"],
        )
    return _expert(
        id_="mom_quality", label="Momentum + Quality", family=FAMILY_PRICE,
        status="unknown" if not mom_ok and quality.get("status") == "unknown" else "neutral",
        eligible=False, score=None,
        thesis="Momentum + Quality", horizon=HORIZON_POSITION,
        evidence=["Needs both price leadership and readable quality"],
        quality_flags=["Thesis expert — does not add a third evidence family"],
    )


def evaluate_row_experts(
    row: Mapping[str, Any],
    *,
    momentum_ranks: Mapping[str, tuple[int, int, float]] | None = None,
) -> list[dict[str, Any]]:
    ranks = momentum_ranks or {}
    xs = _xs_momentum_expert(row, ranks)
    quality = _quality_expert(row)
    experts = [
        _sepa_expert(row),
        xs,
        _rs_expert(row),
        _breakout_expert(row),
        _vcp_expert(row),
        _pullback_expert(row),
        _earnings_expert(row),
        quality,
        _mom_quality_expert(xs, quality),
        _sector_expert(row),
        _volume_expert(row),
        _catalyst_expert(row),
    ]
    return experts


def attach_experts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Score every row against the expert panel. Ranks are cross-sectional in this list."""
    universe = [dict(r) for r in rows]
    mom_ranks = _rank_map(universe, "momentum_5d")
    out: list[dict[str, Any]] = []
    for row in universe:
        painted = dict(row)
        painted["experts"] = evaluate_row_experts(painted, momentum_ranks=mom_ranks)
        out.append(painted)
    return out
