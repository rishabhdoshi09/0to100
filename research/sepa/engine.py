"""Canonical SEPA-001 eligibility — research/backtest only.

Does not place orders, does not change autopilot, does not alter production BUY.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from research.sepa.config import DEFAULT_CONFIG, SepaConfig
from research.sepa.entry import evaluate_entry
from research.sepa.frames import (
    atr,
    ca_status,
    close_series,
    iso_date,
    last_session_iso,
    load_symbol_frame,
    pit_universe,
    slice_as_of,
)
from research.sepa.rs import build_rs_table, lookup_rs
from research.sepa.trend import evaluate_trend
from research.sepa.types import SepaEligibility
from research.sepa.vcp import detect_vcp


def evaluate_sepa_eligibility(
    symbol: str,
    as_of_date,
    *,
    frame=None,
    frames: Mapping[str, Any] | None = None,
    universe: Sequence[str] | None = None,
    rs_table: Mapping[str, Any] | None = None,
    rs_percentile: float | None = None,
    config: SepaConfig | None = None,
    pit_meta: Mapping[str, Any] | None = None,
    buy_zone_above_pct: float | None = None,
) -> SepaEligibility:
    """Is this stock a SEPA-style *trade* at this exact as-of date?"""
    cfg = config or DEFAULT_CONFIG
    sym = str(symbol or "").upper().strip()
    as_of = iso_date(as_of_date)
    meta = dict(pit_meta or {})

    sliced = slice_as_of(frame, as_of) if frame is not None else None
    if sliced is None and frame is None:
        sliced = load_symbol_frame(sym, as_of)

    if "universe_complete" not in meta:
        try:
            u = pit_universe(as_of)
            meta.setdefault("universe_complete", u["universe_complete"])
            meta.setdefault("universe_source", u.get("source") or "")
            meta.setdefault("universe_note", u.get("note") or "")
            if universe is None:
                universe = u.get("symbols")
        except Exception as exc:
            meta.setdefault("universe_complete", False)
            meta.setdefault("universe_note", f"universe unread: {exc}")
    if "ca_complete" not in meta:
        ca = ca_status()
        meta["ca_complete"] = bool(ca.get("ca_complete"))
        meta["ca_note"] = ca.get("note") or ""

    universe_version = str(meta.get("universe_source") or "")
    if meta.get("universe_complete"):
        universe_version = universe_version or "pit_universe"
    else:
        universe_version = universe_version or "survivors_degraded"

    rs_info: dict[str, Any]
    if rs_percentile is not None:
        rs_info = {
            "available": True,
            "percentile": float(rs_percentile),
            "score": None,
            "components": {},
            "n_ranked": None,
            "injected": True,
        }
    else:
        table = rs_table
        if table is None and frames is not None:
            table = build_rs_table(frames, as_of, cfg, universe=universe)
        rs_info = lookup_rs(table, sym) if table is not None else {
            "available": False, "percentile": None, "score": None, "components": {},
        }

    trend = evaluate_trend(sliced, cfg, rs_percentile=rs_info.get("percentile"))
    vcp = detect_vcp(sliced, cfg) if sliced is not None else detect_vcp(None, cfg)
    close = close_series(sliced)
    price = float(close.iloc[-1]) if close is not None and len(close) else None
    atr_val = atr(sliced, cfg.atr_period) if sliced is not None else None
    entry = evaluate_entry(
        price=price, vcp=vcp, atr=atr_val, config=cfg,
        buy_zone_above_pct=buy_zone_above_pct,
    )

    bench = {}
    if sliced is not None:
        try:
            from product.monitor_context import nifty_frame, rs_vs_benchmark
            bench_raw = nifty_frame()
            bench_sliced = slice_as_of(bench_raw, as_of) if bench_raw is not None else None
            bench = rs_vs_benchmark(sliced, bench_sliced)
        except Exception:
            bench = {"available": False}

    good_stock = bool(trend["trend_template_pass"])
    good_setup = bool(vcp.get("detected")) and vcp.get("pivot") is not None
    good_entry = bool(entry.get("entry_valid")) and bool(entry.get("stop_ok"))
    codes: list[str] = []
    reasons: list[str] = []

    if sliced is None:
        codes.append("INSUFFICIENT_HISTORY")
        reasons.append("No official OHLCV through as_of.")
    if not trend["trend_template_pass"]:
        codes.append("TREND_TEMPLATE_FAIL")
        failed = [r.id for r in trend["rules"] if r.passed is not True]
        reasons.append("Trend template not 8/8: " + ", ".join(failed))
    if not rs_info.get("available"):
        codes.append("RS_UNAVAILABLE")
    elif not bool(trend["rules"][-1].passed):
        codes.append("RS_FAIL")
    if not vcp.get("detected"):
        codes.append("VCP_NOT_DETECTED")
        reasons.extend(vcp.get("fail_reasons") or [])
    if vcp.get("pivot") is None:
        codes.append("NO_PIVOT")
        reasons.append("No structurally valid pivot — none manufactured.")
    if entry.get("extended"):
        codes.append("ENTRY_EXTENDED")
        reasons.append(
            f"NO TRADE — INVALID ENTRY: price {price} vs pivot {entry.get('pivot')} "
            f"({entry.get('distance_from_pivot_pct')}% above)."
        )
    elif entry.get("entry_rejection") == "ENTRY_BELOW_PIVOT":
        codes.append("ENTRY_BELOW_PIVOT")
        reasons.append("Price still below the buy-zone — setup may be forming, not a trade.")
    elif entry.get("entry_rejection") == "WIDE_STRUCTURAL_STOP":
        codes.append("WIDE_STRUCTURAL_STOP")
        reasons.append("Structural stop is too far from entry; not tightened artificially.")
    elif entry.get("entry_rejection") and entry.get("entry_rejection") not in {"NO_PIVOT"}:
        codes.append(str(entry.get("entry_rejection")))

    eligible = good_stock and good_setup and good_entry
    pit_safe = bool(meta.get("universe_complete")) and bool(meta.get("ca_complete"))
    if not meta.get("universe_complete"):
        reasons.append("Universe membership is not PIT-complete (survivorship risk).")
    if not meta.get("ca_complete"):
        reasons.append("Corporate-action table missing — prices may contain phantom gaps.")

    if eligible:
        headline = "ELIGIBLE — stock + setup + entry"
    elif good_stock and good_setup and entry.get("extended"):
        headline = "NO TRADE — INVALID ENTRY"
    elif good_stock and good_setup and not good_entry:
        headline = "NO TRADE — INVALID ENTRY"
    elif good_stock and not good_setup:
        headline = "GOOD STOCK — SETUP NOT STRUCTURAL"
    elif not good_stock:
        headline = "NOT STAGE-2 / RS LEADER"
    else:
        headline = "NOT ELIGIBLE"

    seen = []
    for c in codes:
        if c not in seen:
            seen.append(c)

    return SepaEligibility(
        symbol=sym,
        as_of_date=as_of,
        data_timestamp=last_session_iso(sliced),
        eligibility_version=cfg.eligibility_version,
        config_hash=cfg.config_hash(),
        universe_version=universe_version,
        trend_rules=list(trend["rules"]),
        trend_template_pass=bool(trend["trend_template_pass"]),
        structure_pass=bool(trend.get("structure_pass")),
        trend_passed=int(trend["passed"]),
        trend_total=8,
        levels=dict(trend["levels"]),
        rs_score=rs_info.get("score"),
        rs_percentile=rs_info.get("percentile"),
        rs_threshold=cfg.rs_threshold,
        rs_pass=bool(rs_info.get("available") and (rs_info.get("percentile") or -1) >= cfg.rs_threshold),
        rs_components=dict(rs_info.get("components") or {}),
        benchmark_rs=dict(bench or {}),
        setup_type="VCP" if vcp.get("detected") else "",
        vcp_detected=bool(vcp.get("detected")),
        contraction_count=int(vcp.get("contraction_count") or 0),
        contraction_depths=list(vcp.get("depths") or []),
        contraction_dates=list(vcp.get("dates") or []),
        contraction_durations=list(vcp.get("durations") or []),
        base_depth_pct=vcp.get("base_depth_pct"),
        final_contraction_pct=vcp.get("final_depth_pct"),
        tightness=vcp.get("tightness"),
        vol_first=vcp.get("vol_first"),
        vol_final=vcp.get("vol_final"),
        vol_recent_vs_base=vcp.get("vol_recent_vs_base"),
        dry_up_ratio=vcp.get("dry_up_ratio"),
        setup_quality=vcp.get("quality"),
        setup_fail_reasons=list(vcp.get("fail_reasons") or []),
        pivot=entry.get("pivot"),
        pivot_date=entry.get("pivot_date"),
        pivot_type=entry.get("pivot_type"),
        price=price,
        distance_from_pivot_pct=entry.get("distance_from_pivot_pct"),
        buy_zone_low=entry.get("buy_zone_low"),
        buy_zone_high=entry.get("buy_zone_high"),
        entry_valid=bool(entry.get("entry_valid")),
        entry_rejection=entry.get("entry_rejection"),
        extended=bool(entry.get("extended")),
        proposed_entry=entry.get("proposed_entry"),
        structural_stop=entry.get("structural_stop"),
        stop_basis=entry.get("stop_basis"),
        stop_distance_pct=entry.get("stop_distance_pct"),
        atr=entry.get("atr"),
        stop_atr_multiple=entry.get("stop_atr_multiple"),
        stop_ok=bool(entry.get("stop_ok")),
        risk_r=entry.get("risk_r"),
        measured_move=entry.get("measured_move"),
        reward_price=entry.get("reward_price"),
        reward_risk=entry.get("reward_risk"),
        reward_status=str(entry.get("reward_status") or "UNKNOWN"),
        resistance=dict(entry.get("resistance") or {}),
        good_stock=good_stock,
        good_setup=good_setup,
        good_entry=good_entry,
        eligible=eligible,
        rejection_codes=seen,
        reasons=reasons,
        headline=headline,
        pit_safe=pit_safe,
        universe_complete=bool(meta.get("universe_complete")),
        ca_complete=bool(meta.get("ca_complete")),
        research_grade=pit_safe and eligible,
        evidence={
            "near_sepa": bool(trend.get("near_sepa")),
            "vcp_evidence": vcp.get("evidence") or {},
            "buy_zone_above_pct": buy_zone_above_pct if buy_zone_above_pct is not None else cfg.buy_zone_above_pct,
            "atr_wide_diagnostic": bool(entry.get("evidence_atr_wide")),
            "pit": {
                "universe_complete": bool(meta.get("universe_complete")),
                "ca_complete": bool(meta.get("ca_complete")),
                "universe_note": meta.get("universe_note") or "",
                "ca_note": meta.get("ca_note") or "",
            },
            "rs_injected": bool(rs_percentile is not None),
        },
    )
