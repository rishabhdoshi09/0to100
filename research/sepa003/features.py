"""Build one explanatory row from a frozen SepaEligibility + fill path."""
from __future__ import annotations

from typing import Any, Mapping

from research.sepa.ablation import _net
from research.sepa003.constants import FEATURE_SET, era_of, rs_bucket


def _rule_pass(sepa, rid: str) -> bool | None:
    for r in sepa.trend_rules or []:
        ident = r.id if hasattr(r, "id") else (r.get("id") if isinstance(r, dict) else None)
        if ident == rid:
            return r.passed if hasattr(r, "passed") else r.get("passed")
    return None


def _num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:
        return None
    return v


def pack_features(
    *,
    sepa,
    sim: Mapping[str, Any] | None,
    fill_class: str,
    detection_date: str,
    entry_date: str | None,
    regime_detect: Mapping[str, Any],
    regime_entry: Mapping[str, Any],
    sector_ctx: Mapping[str, Any],
    breadth: Mapping[str, Any] | None = None,
    fwd_pct_20: float | None = None,
    ca_censored: bool = False,
    setup_id: str = "",
    source_variant: str = "F",
    is_control: bool = False,
    control_kind: str | None = None,
) -> dict[str, Any]:
    depths = list(sepa.contraction_depths or [])
    first_d = depths[0] if depths else None
    final_d = sepa.final_contraction_pct if sepa.final_contraction_pct is not None else (depths[-1] if depths else None)
    ratio = None
    if first_d not in (None, 0) and final_d is not None:
        ratio = float(final_d) / float(first_d)
    lv = sepa.levels or {}
    price = _num(sepa.price)
    s50 = _num(lv.get("sma50"))
    s150 = _num(lv.get("sma150"))
    s200 = _num(lv.get("sma200"))
    s200p = _num(lv.get("sma200_prev"))
    comps = sepa.rs_components or {}
    entry = _num((sim or {}).get("entry"))
    stop = _num(sepa.structural_stop)
    gap = None
    if entry is not None and price not in (None, 0):
        gap = entry / price - 1.0
    zone_pos = None
    lo, hi = _num(sepa.buy_zone_low), _num(sepa.buy_zone_high)
    if entry is not None and lo is not None and hi is not None and hi != lo:
        zone_pos = (entry - lo) / (hi - lo)
    net = None
    if sim is not None and entry is not None and stop is not None:
        net = _net(float(sim.get("gross_r") or 0.0), entry, stop)
    as_of = entry_date or detection_date
    row = {
        "experiment": "SEPA-003",
        "feature_set": FEATURE_SET,
        "setup_id": setup_id or sepa.setup_id,
        "symbol": sepa.symbol,
        "source_variant": source_variant,
        "is_control": is_control,
        "control_kind": control_kind,
        "detection_date": detection_date,
        "entry_date": entry_date,
        "as_of": as_of,
        "year": as_of[:4],
        "era": era_of(as_of),
        "fill_class": fill_class,
        "ca_censored": bool(ca_censored),
        "left_censored": bool(sepa.left_censored),
        "trend_template_pass": bool(sepa.trend_template_pass),
        "structure_pass": bool(sepa.structure_pass),
        "trend_passed": int(sepa.trend_passed),
        "rule_price_gt_150_200": _rule_pass(sepa, "price_gt_150_200"),
        "rule_sma150_gt_200": _rule_pass(sepa, "sma150_gt_200"),
        "rule_sma200_rising": _rule_pass(sepa, "sma200_rising"),
        "rule_sma50_leads": _rule_pass(sepa, "sma50_leads"),
        "rule_price_gt_sma50": _rule_pass(sepa, "price_gt_sma50"),
        "rule_off_52w_low": _rule_pass(sepa, "off_52w_low"),
        "rule_near_52w_high": _rule_pass(sepa, "near_52w_high"),
        "rule_rs_percentile": _rule_pass(sepa, "rs_percentile"),
        "dist_sma50_pct": None if price is None or not s50 else (price / s50 - 1.0) * 100.0,
        "dist_sma150_pct": None if price is None or not s150 else (price / s150 - 1.0) * 100.0,
        "dist_sma200_pct": None if price is None or not s200 else (price / s200 - 1.0) * 100.0,
        "sma200_slope": None if s200 is None or s200p in (None, 0) else (s200 / s200p - 1.0) * 100.0,
        "dist_52w_high_pct": _num(lv.get("below_high_pct")),
        "dist_52w_low_pct": _num(lv.get("above_low_pct")),
        "rs_percentile": _num(sepa.rs_percentile),
        "rs_score": _num(sepa.rs_score),
        "rs_bucket": rs_bucket(sepa.rs_percentile),
        "rs_r63": _num(comps.get("r63")),
        "rs_r126": _num(comps.get("r126")),
        "rs_r189": _num(comps.get("r189")),
        "rs_r252": _num(comps.get("r252")),
        "benchmark_excess": _num((sepa.benchmark_rs or {}).get("excess_63") or (sepa.benchmark_rs or {}).get("rs")),
        "vcp_detected": bool(sepa.vcp_detected),
        "vcp_state": sepa.vcp_state,
        "contraction_count": int(sepa.contraction_count or 0),
        "contraction_depths": depths,
        "first_contraction_pct": first_d,
        "final_contraction_pct": final_d,
        "final_over_first": ratio,
        "base_depth_pct": _num(sepa.base_depth_pct),
        "base_duration": None,
        "tightness": _num(sepa.tightness),
        "vcp_quality": _num(sepa.setup_quality),
        "dry_up_ratio": _num(sepa.dry_up_ratio),
        "vol_recent_vs_base": _num(sepa.vol_recent_vs_base),
        "pivot": _num(sepa.pivot),
        "distance_from_pivot_pct": _num(sepa.distance_from_pivot_pct),
        "buy_zone_position": zone_pos,
        "breakout_gap_pct": None if gap is None else gap * 100.0,
        "next_open_distance_pct": None if gap is None else gap * 100.0,
        "atr": _num(sepa.atr),
        "stop_distance_pct": _num(sepa.stop_distance_pct),
        "stop_basis": sepa.stop_basis,
        "stop_atr_multiple": _num(sepa.stop_atr_multiple),
        "price": price,
        "turnover_proxy": None,
        "liquidity": None,
        "regime_detection": regime_detect.get("regime"),
        "regime_entry": regime_entry.get("regime"),
        "nifty_trend_state": regime_entry.get("trend_state"),
        "idx_dist_sma50": regime_entry.get("dist_sma50_pct"),
        "idx_dist_sma200": regime_entry.get("dist_sma200_pct"),
        "idx_slope_sma50": regime_entry.get("slope_sma50_pct"),
        "idx_slope_sma200": regime_entry.get("slope_sma200_pct"),
        "idx_ret20": regime_entry.get("ret20"),
        "breadth_pct_above_50": None if not breadth else breadth.get("pct_above_50"),
        "breadth_verdict": None if not breadth else breadth.get("verdict"),
        "sector": sector_ctx.get("sector") or "UNKNOWN",
        "sector_ret": sector_ctx.get("sector_ret"),
        "sector_rs": sector_ctx.get("sector_rs"),
        "sector_rank": sector_ctx.get("sector_rank"),
        "stock_vs_sector": sector_ctx.get("stock_vs_sector"),
        "n_strong_in_group": sector_ctx.get("n_strong_in_group"),
        "n_sector_members": sector_ctx.get("n_sector_members"),
        "net_r": net,
        "gross_r": None if sim is None else _num(sim.get("gross_r")),
        "mae_r": None if sim is None else _num(sim.get("mae_r")),
        "mfe_r": None if sim is None else _num(sim.get("mfe_r")),
        "reached_1r": None if sim is None else bool(sim.get("reached_1r")),
        "reached_2r": None if sim is None else bool(sim.get("reached_2r")),
        "stop_before_1r": None if sim is None else bool(sim.get("stop_before_1r")),
        "hold_sessions": None if sim is None else sim.get("hold_sessions") or sim.get("hold"),
        "failed_breakout": None if sim is None else bool(sim.get("failed_break")),
        "fwd_pct_20": fwd_pct_20,
        "new_hypothesis": True,
        "not_validated_edge": True,
        "confirmation_already_observed": True,
    }
    ev = sepa.evidence or {}
    vcp_ev = ev.get("vcp_evidence") or {}
    if sepa.base_start_date and detection_date:
        try:
            row["base_duration"] = int(
                (pd_ts(detection_date) - pd_ts(sepa.base_start_date)).days
            )
        except Exception:
            row["base_duration"] = vcp_ev.get("base_duration")
    return row


def pd_ts(value):
    import pandas as pd
    return pd.Timestamp(value)
