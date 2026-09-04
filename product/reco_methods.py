"""Multi-method research overlay for Recommendations.

SEPA is one expert. Reco Buy requires two independent *evidence families*
from persisted engines — never a new scanner, never a weighted score soup,
never an LLM money-path.

Honesty:
  • Page-open reads persisted overlays. It does not rescore OHLCV.
  • n < 30 never counts as a Live EV or Case pass.
  • SEPA-only is one family — not a Buy.
  • Tape + RS + momentum collapse into Price Leadership.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.breakout_quality import (
    RSI_HARD,
    has_usable_fundamentals,
    passes_volume_floor,
)
from product.radar_workspace import (
    is_sniper_breakout_candidate,
    load_sepa_overlay_cards,
    merge_fundamental_context,
    merge_sepa_overlay,
)
from product.reco_ensemble import (
    allows_buy as ensemble_allows_buy,
    attach_expert_layer,
    sort_key as ensemble_sort_key,
)

MIN_CONFIRMS_FOR_BUY = 2
SEPA_PASS = 40
RS_PASS_PCTL = 70.0
RS_FAIL_PCTL = 40.0
EV_MIN_N = 30
CASE_MIN_N = 30
CONVICTION_PASS = 75.0

# Weights sum to 1. SEPA is not the majority.
METHOD_WEIGHTS: dict[str, float] = {
    "tape": 0.18,
    "sepa": 0.12,
    "funds": 0.18,
    "trend": 0.12,
    "rs": 0.12,
    "ev": 0.12,
    "conviction": 0.08,
    "case": 0.04,
    "sector": 0.04,
}

METHOD_LABELS: dict[str, str] = {
    "tape": "Tape",
    "sepa": "SEPA",
    "funds": "Funds",
    "trend": "Trend",
    "rs": "RS",
    "ev": "Live EV",
    "conviction": "Conviction",
    "case": "Case memory",
    "sector": "Sector",
}


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


def _method(id_: str, status: str, detail: str, points: float | None) -> dict[str, Any]:
    return {
        "id": id_,
        "label": METHOD_LABELS[id_],
        "status": status,
        "detail": detail,
        "points": None if points is None else round(float(points), 1),
    }


def _tape_method(row: Mapping[str, Any]) -> dict[str, Any]:
    if bool(row.get("chase_risk")):
        return _method("tape", "fail", "Chase / extension — tape rejects", 0.0)
    rsi = _f(row.get("rsi"))
    if rsi is not None and rsi > RSI_HARD:
        return _method("tape", "fail", f"RSI {rsi:.0f} > {RSI_HARD:.0f} blow-off", 0.0)
    vol_known = _f(row.get("volume_ratio"))
    if vol_known is not None and vol_known > 0 and vol_known < 0.7:
        return _method("tape", "fail", f"Volume {vol_known:.2f}× below 0.7× floor", 0.0)
    sniper = is_sniper_breakout_candidate(row)
    grade = str(row.get("breakout_grade") or "").upper()
    vol_ok = passes_volume_floor(row)
    if sniper or (grade in {"A", "B"} and vol_ok):
        pts = 90.0 if sniper else 80.0
        if grade == "A":
            pts = min(100.0, pts + 8)
        return _method("tape", "pass", f"Sniper/grade {grade or '—'} with volume floor", pts)
    status = str(row.get("status") or "")
    if vol_ok and status in {"Ready to trade", "Watch for breakout"}:
        return _method("tape", "pass", f"{status} with readable volume", 70.0)
    if vol_known is None or vol_known <= 0:
        return _method("tape", "unknown", "Volume not on file — tape not confirmed", None)
    return _method("tape", "unknown", "No sniper/grade/ready tape confirm", None)


def _sepa_method(row: Mapping[str, Any]) -> dict[str, Any]:
    score = _f(row.get("sepa_score"))
    if score is None:
        return _method("sepa", "unknown", "No persisted SEPA overlay for this name", None)
    if score >= SEPA_PASS:
        return _method("sepa", "pass", f"SEPA overlay {score:.0f}/100", min(100.0, score))
    return _method("sepa", "fail", f"SEPA overlay {score:.0f}/100 below {SEPA_PASS}", score)


def _funds_method(row: Mapping[str, Any]) -> dict[str, Any]:
    cls = str(row.get("classification") or "")
    if cls == "AVOID_REVIEW":
        return _method("funds", "fail", "AVOID_REVIEW — fundamentals reject", 0.0)
    cov = _f(row.get("fundamental_coverage"))
    if has_usable_fundamentals(row):
        fund = _f(row.get("fundamental_score")) or 70.0
        return _method(
            "funds", "pass",
            f"{cls.replace('_', ' ') or 'Funds'} · coverage {(cov or 0) * 100:.0f}%",
            min(100.0, fund),
        )
    if cov is not None and 0 < cov < 0.50 and cls:
        return _method("funds", "fail", f"Coverage {cov:.0%} < 50%", 0.0)
    if not cls and cov is None and row.get("fundamental_score") is None:
        return _method("funds", "unknown", "No long-term fundamental row on file", None)
    return _method("funds", "unknown", "Fundamentals present but not a quality class with ≥50% coverage", None)


def _trend_method(row: Mapping[str, Any]) -> dict[str, Any]:
    n_struct = row.get("n_structure_passed")
    try:
        n_struct_i = int(n_struct) if n_struct is not None and n_struct != "" else None
    except (TypeError, ValueError):
        n_struct_i = None
    if n_struct_i is not None and n_struct_i >= 3:
        return _method("trend", "pass", f"Feature-001 structure {n_struct_i} checks passed", min(100.0, 40.0 + n_struct_i * 12))
    above50 = row.get("above_sma50")
    above200 = row.get("above_sma200")
    if above50 is False and above200 is False:
        return _method("trend", "fail", "Below SMA50 and SMA200", 0.0)
    if above50 is True and above200 is True:
        return _method("trend", "pass", "Above SMA50 and SMA200", 85.0)
    if above50 is True or above200 is True:
        which = "SMA50" if above50 is True else "SMA200"
        return _method("trend", "pass", f"Above {which}", 70.0)
    if n_struct_i is not None and n_struct_i < 3:
        return _method("trend", "fail", f"Structure only {n_struct_i} checks", 20.0)
    return _method("trend", "unknown", "No SMA flags or feature-001 structure on file", None)


def _rs_method(row: Mapping[str, Any]) -> dict[str, Any]:
    pctl = _f(row.get("rs_percentile"))
    vs = _f(row.get("rs_vs_nifty") or row.get("rs_vs_nifty_20d") or row.get("rs_score"))
    if pctl is not None:
        if pctl >= RS_PASS_PCTL:
            return _method("rs", "pass", f"RS percentile {pctl:.0f}", pctl)
        if pctl < RS_FAIL_PCTL:
            return _method("rs", "fail", f"RS percentile {pctl:.0f} (laggard)", pctl)
        return _method("rs", "unknown", f"RS percentile {pctl:.0f} is mid-pack, not a confirm", pctl)
    if vs is not None:
        if vs > 0:
            return _method("rs", "pass", f"RS vs Nifty {vs:+.1f}", min(100.0, 55.0 + vs))
        if vs < 0:
            return _method("rs", "fail", f"RS vs Nifty {vs:+.1f}", max(0.0, 50.0 + vs))
    return _method("rs", "unknown", "No RS percentile or vs-Nifty print on file", None)


def _ev_method(row: Mapping[str, Any]) -> dict[str, Any]:
    n = _f(row.get("ev_n"))
    lb = _f(row.get("ev_lb_pct"))
    if n is None or n < EV_MIN_N:
        return _method("ev", "unknown", "Live EV needs ≥30 comparable outcomes", None)
    if lb is None:
        return _method("ev", "unknown", f"n={n:.0f} but conservative EV missing", None)
    if lb > 0:
        return _method("ev", "pass", f"Conservative EV {lb:+.1f}% (n={n:.0f})", min(100.0, 50.0 + lb * 8))
    return _method("ev", "fail", f"Conservative EV {lb:+.1f}% (n={n:.0f})", 0.0)


def _conviction_method(row: Mapping[str, Any]) -> dict[str, Any]:
    cls = str(row.get("conviction_class") or "")
    score = _f(row.get("conviction_score"))
    if cls == "WAIT_FOR_PULLBACK":
        return _method("conviction", "fail", "Conviction class WAIT_FOR_PULLBACK", 0.0)
    if cls == "HIGH_CONVICTION" or (score is not None and score >= CONVICTION_PASS):
        return _method("conviction", "pass", f"Setup quality {score:.0f}/100" if score is not None else "HIGH_CONVICTION", score or 80.0)
    if score is None and not cls:
        return _method("conviction", "unknown", "Conviction shortlist not attached", None)
    return _method("conviction", "unknown", f"Conviction {cls or 'unscored'} — not high", score)


def _case_method(row: Mapping[str, Any]) -> dict[str, Any]:
    n = _f(row.get("case_n_similar"))
    if n is None:
        return _method("case", "unknown", "Case memory not attached", None)
    if n < CASE_MIN_N:
        return _method("case", "unknown", f"{n:.0f} similar cases — not proven (<{CASE_MIN_N})", None)
    exp = _f(row.get("case_expectancy_r"))
    if exp is not None and exp > 0:
        return _method("case", "pass", f"{n:.0f} similar · expectancy {exp:+.2f}R", min(100.0, 50.0 + exp * 20))
    if exp is not None and exp < 0:
        return _method("case", "fail", f"{n:.0f} similar · expectancy {exp:+.2f}R", 0.0)
    proven = bool(row.get("case_proven"))
    if proven:
        return _method("case", "pass", f"{n:.0f} similar cases remembered", 70.0)
    return _method("case", "unknown", f"{n:.0f} similar but expectancy unread", None)


def _sector_method(row: Mapping[str, Any]) -> dict[str, Any]:
    if bool(row.get("sector_laggard")):
        return _method("sector", "fail", f"Lagging sector: {row.get('sector') or '—'}", 0.0)
    score = _f(row.get("sector_leadership_score"))
    label = str(row.get("sector_leadership_label") or "Sector")
    if score is not None and score >= 70:
        return _method("sector", "pass", f"{label} {score:.0f} · {row.get('sector') or '—'}", min(100.0, score))
    if bool(row.get("sector_leader")):
        return _method("sector", "pass", f"Leading sector: {row.get('sector') or '—'}", 80.0)
    if score is not None and score < 40:
        return _method("sector", "fail", f"{label} {score:.0f} · {row.get('sector') or '—'}", score)
    if score is not None:
        return _method("sector", "unknown", f"{label} {score:.0f} is mid-pack, ranking only", score)
    return _method("sector", "unknown", "Sector pack not in current leaders/laggards", None)


_EVALUATORS = (
    _tape_method, _sepa_method, _funds_method, _trend_method, _rs_method,
    _ev_method, _conviction_method, _case_method, _sector_method,
)


def score_methods(row: Mapping[str, Any]) -> dict[str, Any]:
    """Independent method panel. Missing inputs stay unknown (composite 0 for that slot)."""
    methods = [fn(row) for fn in _EVALUATORS]
    confirms = [m for m in methods if m["status"] == "pass"]
    fails = [m for m in methods if m["status"] == "fail"]
    known = [m for m in methods if m["status"] != "unknown"]
    composite = 0.0
    for item in methods:
        weight = METHOD_WEIGHTS[item["id"]]
        pts = item["points"]
        if pts is None:
            continue
        composite += weight * max(0.0, min(100.0, float(pts)))
    return {
        "methods": methods,
        "method_confirms": len(confirms),
        "method_fails": len(fails),
        "method_known": len(known),
        "quality_score": round(composite, 2),
        "method_line": (
            " + ".join(m["label"] for m in confirms)
            if confirms else "No independent method confirmed yet"
        ),
    }


def allows_buy(row: Mapping[str, Any]) -> bool:
    """Buy needs two independent *families*, a why-now, and a ready entry.

    Method chips still exist for the evidence panel. The money-path gate is
    the mixture-of-experts ensemble, not a weighted SEPA/momentum soup.
    """
    if row.get("allows_recommend") is not None:
        return bool(row.get("allows_recommend"))
    if row.get("experts") is not None:
        return ensemble_allows_buy(row)
    painted = attach_expert_layer([dict(row)])
    return ensemble_allows_buy(painted[0]) if painted else False


def attach_method_scores(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.update(score_methods(out))
    return out


def _shadow_by_symbol(symbols: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Latest FEATURE-002 observation per symbol. Empty if the ledger is missing."""
    want = sorted({str(s).upper() for s in symbols if str(s).strip()})
    if not want:
        return {}
    try:
        from research.feature002.constants import DB_PATH
    except Exception:
        return {}
    path = DB_PATH
    if not path.exists():
        return {}
    import sqlite3
    out: dict[str, dict[str, Any]] = {}
    try:
        con = sqlite3.connect(str(path), timeout=2.0)
        con.row_factory = sqlite3.Row
        try:
            chunk = 400
            for i in range(0, len(want), chunk):
                part = want[i:i + chunk]
                marks = ",".join("?" * len(part))
                rows = con.execute(
                    f"""
                    SELECT o.symbol, o.rs_percentile, o.rs_score, o.n_structure_passed,
                           o.combined_shadow_rank, o.r3_score, o.session_date
                    FROM observations o
                    INNER JOIN (
                        SELECT symbol, MAX(session_date) AS d
                        FROM observations
                        WHERE symbol IN ({marks})
                        GROUP BY symbol
                    ) t ON t.symbol = o.symbol AND t.d = o.session_date
                    """,
                    part,
                ).fetchall()
                for row in rows:
                    out[str(row["symbol"]).upper()] = dict(row)
        finally:
            con.close()
    except Exception:
        return {}
    return out


def _conviction_by_symbol(scan_rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    try:
        from product.conviction import build_conviction_shortlist
        from product.market_view import build_market_view
        regime: dict[str, Any] = {}
        try:
            from core import regime_engine as re
            cached = (re._CACHE or {}).get("regime_state")
            if cached is not None:
                regime = {
                    "regime_score": getattr(cached, "regime_score", None),
                    "risk_mode": getattr(cached, "risk_mode", None),
                    "breakout_environment": getattr(cached, "breakout_environment", None),
                    "breadth_label": getattr(cached, "breadth_label", None),
                    "leading_sectors": getattr(cached, "leading_sectors", ()) or (),
                    "lagging_sectors": getattr(cached, "lagging_sectors", ()) or (),
                }
        except Exception:
            regime = {}
        view = build_market_view(regime)
        payload = {"records": [dict(r) for r in scan_rows]}
        rows = build_conviction_shortlist(payload, view)
        return {str(r.get("symbol") or "").upper(): r for r in rows if r.get("symbol")}
    except Exception:
        return {}


def _sector_sets() -> tuple[set[str], set[str]]:
    try:
        from core import regime_engine as re
        cached = (re._CACHE or {}).get("regime_state")
        leaders = {str(x).lower() for x in (getattr(cached, "leading_sectors", ()) or ())}
        laggards = {str(x).lower() for x in (getattr(cached, "lagging_sectors", ()) or ())}
        return leaders, laggards
    except Exception:
        return set(), set()


def _case_fields(row: Mapping[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    try:
        from product.case_memory import primary_setup, setup_memory
        setup = primary_setup(row)
        if setup not in cache:
            cache[setup] = dict(setup_memory(setup) or {})
        mem = cache[setup]
        return {
            "case_n_similar": mem.get("n") or 0,
            "case_expectancy_r": mem.get("expectancy_r"),
            "case_proven": int(mem.get("n") or 0) >= CASE_MIN_N,
        }
    except Exception:
        return {}


def attach_research_overlays(
    scan_rows: Sequence[Mapping[str, Any]],
    lt_rows: Sequence[Mapping[str, Any]],
    *,
    scanned_at: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Merge persisted SEPA / funds / shadow / conviction / case onto rows, then score."""
    fund_by = {
        str(r.get("symbol") or "").upper(): r
        for r in lt_rows
        if str(r.get("symbol") or "")
    }
    sepa_cards = load_sepa_overlay_cards(scanned_at)
    sepa_by = {
        str(c.get("symbol") or "").upper(): c
        for c in sepa_cards
        if c.get("symbol")
    }
    symbols = [
        str(r.get("symbol") or "").upper()
        for r in (*scan_rows, *lt_rows)
        if r.get("symbol")
    ]
    shadow_by = _shadow_by_symbol(symbols)
    leaders, laggards = _sector_sets()
    try:
        from product.sector_leadership import board_from_rows
        sector_board = board_from_rows(scan_rows, leaders, laggards)
    except Exception:
        sector_board = {}
    conv_by = _conviction_by_symbol(scan_rows)
    case_cache: dict[str, dict[str, Any]] = {}

    def _paint(row: Mapping[str, Any]) -> dict[str, Any]:
        out = merge_sepa_overlay(merge_fundamental_context(row, fund_by), sepa_by)
        sym = str(out.get("symbol") or "").upper()
        shadow = shadow_by.get(sym) or {}
        for key in ("rs_percentile", "rs_score", "n_structure_passed", "combined_shadow_rank", "r3_score"):
            if out.get(key) is None and shadow.get(key) is not None:
                out[key] = shadow.get(key)
        conv = conv_by.get(sym)
        if conv:
            out.setdefault("conviction_score", conv.get("conviction_score"))
            out.setdefault("conviction_class", conv.get("classification"))
        sector = str(out.get("sector") or "").lower()
        out["sector_leader"] = bool(sector and sector in leaders)
        out["sector_laggard"] = bool(sector and sector in laggards)
        try:
            from product.sector_leadership import attach_to_row
            out.update(attach_to_row(out, sector_board))
        except Exception:
            pass
        out.update(_case_fields(out, case_cache))
        return attach_method_scores(out)

    painted_scan = attach_expert_layer([_paint(r) for r in scan_rows])
    painted_lt = attach_expert_layer([_paint(r) for r in lt_rows])
    return painted_scan, painted_lt


def sort_key(card: Mapping[str, Any]) -> tuple:
    """Tier, then independent families, then scanner score — not a weighted soup."""
    return ensemble_sort_key(card)
