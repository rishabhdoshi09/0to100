"""SEPA-001R2.1 ablation — daily PIT, causal CA segments, session embargo.

Research only. Does not place orders. Does not change production BUY.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from research.sepa.ablation import _net, _scanner_sim
from research.sepa.ablation_r import (
    SimRowR,
    sepa_fill_sim,
    summarize_r,
    _nifty_regimes,
    _regime_at,
    _sector,
)
from research.sepa.ca_audit import CATimeline, build_timeline
from research.sepa.config import R2_CONFIG, SepaConfig
from research.sepa.embargo import attach_session_path, calendar_day_embargo_until, session_embargo_blocks
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.entry import FILL_CA_CENSORED, FILL_EXTENDED, FILL_GAP_THROUGH
from research.sepa.frames import iso_date
from research.sepa.gates import deployment_eligible, statistical_gate
from research.sepa.integrity import research_integrity_report
from research.sepa.lifecycle import PersistentSetupLedger
from research.sepa.scanner_research import research_scanner_analyze, scanner_signal_ok
from research.sepa.signal_study import forward_path_study, summarize_signal_study
from research.sepa.types import SepaEligibility
from research.sepa.universe_pit import FastInvestable

_OUT = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "SEPA-001R2"

SNAPSHOT_FUNNEL_KEYS = (
    "candidates",
    "investable",
    "stage2",
    "rs_pass",
    "vcp_detected",
    "pivot_defined",
    "entry_ready",
)

UNIQUE_FUNNEL_KEYS = (
    "vcp_detected",
    "valid_pivot",
    "entry_ready",
    "valid_fill",
    "gap_through",
    "observed_extended",
    "left_censored",
    "ca_censored",
    "stop_too_wide",
    "expired_failed",
    "pivot_retest",
)

# Predeclared in SEPA_001R2_VALIDATION_PROTOCOL.md — do not move after seeing results.
DEV_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2024-12-31"
CONF_START = "2025-01-01"
CONF_END = "2026-08-21"


def _session_calendar(frames: Mapping[str, pd.DataFrame]) -> list[pd.Timestamp]:
    acc = set()
    for df in frames.values():
        if df is None or len(df) == 0:
            continue
        acc.update(pd.DatetimeIndex(df.index).tz_localize(None).normalize())
    return sorted(acc)


def block_of(as_of: str) -> str:
    if as_of <= DEV_END:
        return "development"
    if VAL_START <= as_of <= VAL_END:
        return "validation"
    if as_of >= CONF_START:
        return "confirmation"
    return "other"


def observe_lifecycle_daily() -> bool:
    """E/F lifecycle exists every session. Scanner does not create the setup."""
    return True


def attempt_e_entry(*, scanner_ok: bool, structure_pass: bool, rs_pass: bool, vcp_detected: bool) -> bool:
    """E fill attempt only when D gates AND the scanner qualify that session."""
    return bool(scanner_ok and structure_pass and rs_pass and vcp_detected)


def run_ablation_r2(
    *,
    frames: Mapping[str, pd.DataFrame],
    config: SepaConfig | None = None,
    horizon: int = 20,
    warmup_sessions: int = 252,
    min_sessions: int = 260,
    min_price: float = 20.0,
    min_turnover: float = 5_000_000.0,
    top_n: int | None = None,
    variants: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G"),
    date_step: int = 1,
    scanner_step: int = 1,
    quarantined: set[str] | None = None,
    ca_timeline: CATimeline | None = None,
    integrity: dict[str, Any] | None = None,
    rs_threshold: float | None = None,
    scanner_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Date-major PIT runner. Canonical observation is every session."""
    cfg = config or R2_CONFIG
    if rs_threshold is not None:
        cfg = replace(cfg, rs_threshold=float(rs_threshold))
    calendar = _session_calendar(frames)
    if len(calendar) < warmup_sessions + horizon + 40:
        return {
            "experiment": "SEPA-001R2",
            "error": "insufficient_calendar",
            "n_sessions": len(calendar),
            "warmup_sessions": warmup_sessions,
        }
    eval_dates = calendar[warmup_sessions: len(calendar) - horizon: max(1, int(date_step))]
    evaluation_start = iso_date(eval_dates[0])
    evaluation_end = iso_date(eval_dates[-1])
    timeline = ca_timeline if ca_timeline is not None else CATimeline([])
    timeline.annotate_calendar(calendar)
    # Legacy static set is recorded, never applied as a historical membership filter.
    static_q = {s.upper() for s in (quarantined or set())} or {
        s for s in timeline.by_symbol
    }
    fast = FastInvestable(frames)
    versions = {
        "eligibility_version": cfg.eligibility_version,
        "vcp_version": cfg.vcp_version,
        "pivot_version": cfg.pivot_version,
    }
    ledgers = {v: PersistentSetupLedger(versions=versions) for v in variants if v in ("E", "F")}
    need_scanner = any(v in variants for v in ("A", "B", "C", "D", "E"))
    scanner = None
    analyze = scanner_fn
    if need_scanner and analyze is None:
        from research.sepa.scanner_research import make_production_scanner
        scanner = make_production_scanner()
        analyze = lambda sym, hist, _s=scanner: research_scanner_analyze(_s, sym, hist)

    regimes = _nifty_regimes()
    rows: dict[str, list[SimRowR]] = {v: [] for v in variants if v != "G"}
    rows_raw: dict[str, list[SimRowR]] = {v: [] for v in variants if v != "G"}
    g_rows: list[dict[str, Any]] = []
    g_rows_raw: list[dict[str, Any]] = []
    last_exit = {v: {} for v in variants}
    last_exit_calendar = {v: {} for v in variants}
    fill_counts = {v: defaultdict(int) for v in variants}
    funnel_snap = {k: 0 for k in SNAPSHOT_FUNNEL_KEYS}
    funnel_unique = {k: 0 for k in UNIQUE_FUNNEL_KEYS}
    seen_unique: dict[str, set[str]] = defaultdict(set)
    yearly_universe: dict[str, dict[str, int]] = {}
    setups: list[dict[str, Any]] = []
    rs_bucket_fwd: dict[str, list[float]] = defaultdict(list)
    n_as_of = 0
    last_snap_meta = None
    diagnostics = {
        "static_quarantine_false_removals": 0,
        "scanner_step5_would_skip_dates": 0,
        "scanner_step5_missed_a": 0,
        "scanner_step5_missed_e_entry_ready": 0,
        "embargo_calendar_disagreements": 0,
        "ca_censored_outcomes": 0,
        "date_step": int(date_step),
        "scanner_step": int(scanner_step),
        "canonical_daily": int(date_step) == 1 and int(scanner_step) == 1,
    }

    def _stamp_row(variant: str, sim: dict[str, Any], sepa: SepaEligibility,
                   as_of: str, fill_class: str, snap_hash: str, snap_src: str,
                   *, raw: bool) -> SimRowR:
        scanner_fill = variant in ("A", "B", "C", "D")
        stop_px = float(sim.get("stop") or sepa.structural_stop or sim.get("entry", 1) * 0.95)
        if scanner_fill and sim.get("scan_stop") is not None:
            stop_px = float(sim["scan_stop"])
        net = _net(sim["gross_r"], sim["entry"], stop_px)
        ext = None
        if sepa.pivot:
            ext = (float(sim["entry"]) / float(sepa.pivot) - 1.0) * 100.0
        return SimRowR(
            variant=variant, symbol=sepa.symbol, as_of=as_of,
            entry=float(sim["entry"]), stop=stop_px,
            net_r=float(net), gross_r=float(sim["gross_r"]),
            hold=int(sim.get("hold_sessions") or sim.get("hold") or 0),
            outcome=str(sim["outcome"]),
            mae_r=float(sim["mae_r"]), mfe_r=float(sim["mfe_r"]),
            reached_1r=bool(sim["reached_1r"]), reached_2r=bool(sim["reached_2r"]),
            stop_before_1r=bool(sim["stop_before_1r"]),
            failed_break=bool(sim["failed_break"]),
            year=as_of[:4], sector=_sector(sepa.symbol),
            regime=_regime_at(regimes, as_of),
            setup_id=sepa.setup_id, fill_class=fill_class,
            extension_at_fill=None if ext is None else round(ext, 4),
            vcp_state=sepa.vcp_state,
            rs_percentile=sepa.rs_percentile,
            unique_setup=not raw,
            entry_date=str(sim.get("entry_date") or ""),
            exit_date=str(sim.get("exit_date") or ""),
            entry_index=int(sim.get("entry_index") or 0),
            exit_index=int(sim.get("exit_index") or 0),
            hold_sessions=int(sim.get("hold_sessions") or sim.get("hold") or 0),
            universe_date=as_of,
            membership_hash=snap_hash,
            membership_source=snap_src,
        )

    def _record(variant: str, sim: dict[str, Any] | None, sepa: SepaEligibility,
                as_of: str, fill_class: str, fwd, snap_hash: str, snap_src: str,
                scan_hit=None):
        if sim is None:
            fill_counts[variant][fill_class] += 1
            return
        sim = attach_session_path(dict(sim), fwd, as_of=as_of)
        if scan_hit is not None:
            sim["scan_stop"] = float(scan_hit.stop)
            sim["stop"] = float(scan_hit.stop)
        exit_s = str(sim.get("exit_date") or as_of)
        if timeline.horizon_crosses(sepa.symbol, as_of, exit_s):
            fill_counts[variant][FILL_CA_CENSORED] += 1
            diagnostics["ca_censored_outcomes"] += 1
            sid = sepa.setup_id or f"{sepa.symbol}:{as_of}"
            if sid not in seen_unique["ca_censored"]:
                seen_unique["ca_censored"].add(sid)
                funnel_unique["ca_censored"] += 1
            return
        raw_row = _stamp_row(variant, sim, sepa, as_of, fill_class, snap_hash, snap_src, raw=True)
        rows_raw[variant].append(raw_row)
        cal_until = calendar_day_embargo_until(as_of, int(sim.get("hold_sessions") or sim.get("hold") or 1))
        blocked = session_embargo_blocks(as_of=as_of, last_exit_session=last_exit[variant].get(sepa.symbol))
        blocked_cal = session_embargo_blocks(as_of=as_of, last_exit_session=last_exit_calendar[variant].get(sepa.symbol))
        if blocked != blocked_cal:
            diagnostics["embargo_calendar_disagreements"] += 1
        if blocked:
            return
        last_exit[variant][sepa.symbol] = exit_s
        last_exit_calendar[variant][sepa.symbol] = cal_until
        fill_counts[variant][fill_class] += 1
        rows[variant].append(_stamp_row(variant, sim, sepa, as_of, fill_class, snap_hash, snap_src, raw=False))

    integ = integrity if integrity is not None else research_integrity_report(
        frames=frames, as_of=eval_dates[0], exhaustive=True,
    )
    ca = (integ.get("ca_integrity") if isinstance(integ, dict) else None) or {}

    for di, as_ts in enumerate(eval_dates):
        as_of = iso_date(as_ts)
        n_as_of += 1
        # Truthful as-of membership: names with a bar ≤ as_of. Source is inferred.
        snap = fast.snapshot(
            as_of, min_price=min_price, min_turnover=min_turnover,
            min_sessions=min_sessions, ca_timeline=timeline, top_n=top_n,
            source="bhav_inferred",
        )
        last_snap_meta = snap.to_meta()
        year = as_of[:4]
        yu = yearly_universe.setdefault(year, {"as_of_points": 0, "investable_sum": 0,
                                               "candidates_sum": 0, "exclusions": defaultdict(int)})
        yu["as_of_points"] += 1
        yu["investable_sum"] += len(snap.investable)
        yu["candidates_sum"] += len(snap.candidates)
        for k, v in snap.exclusions.items():
            yu["exclusions"][k] += int(v)

        for sym in snap.investable:
            if timeline.would_static_quarantine(sym):
                diagnostics["static_quarantine_false_removals"] += 1

        funnel_snap["candidates"] += len(snap.candidates)
        funnel_snap["investable"] += len(snap.investable)
        s2 = rs_n = vcp_n = 0
        rs_table = fast.rs_table(as_of, snap.investable, cfg, timeline=timeline)
        scan_today = bool(need_scanner) and int(scanner_step) < 90
        if need_scanner and int(scanner_step) not in (0, 1) and int(scanner_step) < 90:
            diagnostics["scanner_step5_would_skip_dates"] += int(di % 5 != 0)

        pit_meta_asof = {
            "universe_complete": False,
            "universe_source": "bhav_inferred",
            "universe_note": "as-of membership inferred from official bars ≤ as_of; not an official listing archive",
            "research_grade": False,
            "ca_complete": bool(ca.get("ca_complete")),
            "ca_note": ca.get("note") or "",
            "ca_verified": bool(ca.get("verified")),
            "ca_n_events": int(ca.get("n_events") or 0),
        }

        for sym in snap.investable:
            hist, fwd = fast.hist_fwd(sym, as_of, horizon, timeline=timeline)
            if hist is None or len(hist) < min(80, min_sessions):
                continue
            meta = {**pit_meta_asof, **snap.to_meta(sym)}
            sepa = evaluate_sepa_eligibility(
                sym, as_of, frame=hist, frames=frames, universe=snap.investable,
                rs_table=rs_table, config=cfg, pit_meta=meta, compute_vcp=False,
            )
            if sepa.structure_pass:
                sepa = evaluate_sepa_eligibility(
                    sym, as_of, frame=hist, frames=frames, universe=snap.investable,
                    rs_table=rs_table, config=cfg, pit_meta=meta, compute_vcp=True,
                )
            if sepa.rs_percentile is not None and fwd is not None and len(fwd) >= horizon:
                if not timeline.horizon_crosses(sym, as_of, iso_date(fwd.index[min(horizon, len(fwd)) - 1])):
                    r0 = float(hist["close"].iloc[-1])
                    r1 = float(fwd["close"].iloc[-1])
                    if r0 > 0:
                        fwd_pct = r1 / r0 - 1.0
                        pct = float(sepa.rs_percentile)
                        if 50 <= pct < 70:
                            rs_bucket_fwd["50-69"].append(fwd_pct)
                        elif 70 <= pct < 80:
                            rs_bucket_fwd["70-79"].append(fwd_pct)
                        elif 80 <= pct < 90:
                            rs_bucket_fwd["80-89"].append(fwd_pct)
                        elif 90 <= pct < 95:
                            rs_bucket_fwd["90-94"].append(fwd_pct)
                        elif pct >= 95:
                            rs_bucket_fwd["95-99"].append(fwd_pct)

            if sepa.structure_pass:
                funnel_snap["stage2"] += 1
                s2 += 1
            if sepa.structure_pass and sepa.rs_pass:
                funnel_snap["rs_pass"] += 1
                rs_n += 1
            if sepa.vcp_detected:
                funnel_snap["vcp_detected"] += 1
                vcp_n += 1
            if sepa.vcp_state in {"PIVOT_DEFINED", "ENTRY_READY"} or sepa.pivot:
                funnel_snap["pivot_defined"] += 1
            if sepa.entry_valid:
                funnel_snap["entry_ready"] += 1

            scan_hit = None
            scanner_ok = False
            if scan_today and analyze is not None:
                try:
                    scan_hit = analyze(sym, hist)
                except Exception:
                    scan_hit = None
                scanner_ok = scanner_signal_ok(scan_hit)
            step5_skip = need_scanner and (di % 5 != 0)

            if "A" in variants and scanner_ok and scan_hit is not None and fwd is not None:
                if step5_skip:
                    diagnostics["scanner_step5_missed_a"] += 1
                sim = _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                   float(scan_hit.target), horizon)
                _record("A", sim, sepa, as_of, "SCANNER", fwd, snap.membership_hash, snap.source, scan_hit)
            if "B" in variants and scanner_ok and sepa.structure_pass and scan_hit is not None and fwd is not None:
                sim = _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                   float(scan_hit.target), horizon)
                _record("B", sim, sepa, as_of, "SCANNER", fwd, snap.membership_hash, snap.source, scan_hit)
            if "C" in variants and scanner_ok and sepa.structure_pass and sepa.rs_pass and scan_hit is not None and fwd is not None:
                sim = _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                   float(scan_hit.target), horizon)
                _record("C", sim, sepa, as_of, "SCANNER", fwd, snap.membership_hash, snap.source, scan_hit)
            if "D" in variants and scanner_ok and sepa.structure_pass and sepa.rs_pass and sepa.vcp_detected and scan_hit is not None and fwd is not None:
                sim = _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                   float(scan_hit.target), horizon)
                _record("D", sim, sepa, as_of, "SCANNER", fwd, snap.membership_hash, snap.source, scan_hit)

            if "G" in variants and sepa.structure_pass and sepa.rs_pass and fwd is not None and len(fwd) >= 1:
                packed = forward_path_study(fwd)
                if packed is not None:
                    packed["symbol"] = sym
                    packed["as_of"] = as_of
                    packed["year"] = as_of[:4]
                    packed["sector"] = _sector(sym)
                    packed["regime"] = _regime_at(regimes, as_of)
                    packed["rs_percentile"] = sepa.rs_percentile
                    packed["universe_date"] = as_of
                    packed["membership_hash"] = snap.membership_hash
                    packed["membership_source"] = snap.source
                    exit_s = packed["exit_date"]
                    if timeline.horizon_crosses(sym, as_of, exit_s):
                        packed["ca_censored"] = True
                        diagnostics["ca_censored_outcomes"] += 1
                        fill_counts["G"][FILL_CA_CENSORED] += 1
                        sid = f"G:{sym}:{as_of}"
                        if sid not in seen_unique["ca_censored"]:
                            seen_unique["ca_censored"].add(sid)
                            funnel_unique["ca_censored"] += 1
                    else:
                        packed["ca_censored"] = False
                        g_rows_raw.append(packed)
                        blocked = session_embargo_blocks(
                            as_of=as_of, last_exit_session=last_exit["G"].get(sym),
                        )
                        cal_until = calendar_day_embargo_until(as_of, int(packed["hold_sessions"]))
                        blocked_cal = session_embargo_blocks(
                            as_of=as_of, last_exit_session=last_exit_calendar["G"].get(sym),
                        )
                        if blocked != blocked_cal:
                            diagnostics["embargo_calendar_disagreements"] += 1
                        if not blocked:
                            last_exit["G"][sym] = exit_s
                            last_exit_calendar["G"][sym] = cal_until
                            g_rows.append(packed)
                            fill_counts["G"]["SIGNAL_OPEN"] += 1

            def _unique(stage: str, sid: str) -> None:
                if sid and sid not in seen_unique[stage]:
                    seen_unique[stage].add(sid)
                    funnel_unique[stage] += 1

            def _sepa_lifecycle(variant: str, *, e_opportunity: bool | None = None):
                if variant not in ledgers:
                    return
                if not sepa.pivot and not sepa.base_start_date:
                    return
                ledger = ledgers[variant]
                rec = ledger.observe(
                    symbol=sym, vcp={
                        "base_start_date": sepa.base_start_date,
                        "pivot": sepa.pivot,
                        "stop": sepa.structural_stop,
                        "detected": sepa.vcp_detected,
                        "state": sepa.vcp_state,
                        "dates": sepa.contraction_dates,
                        "contraction_count": sepa.contraction_count,
                        "vcp_knowable_date": sepa.vcp_knowable_date,
                        "evidence": sepa.evidence.get("vcp_evidence") or {},
                    },
                    as_of=as_of,
                    evaluation_start=evaluation_start,
                    price=sepa.price,
                    zone_hi=sepa.buy_zone_high,
                    in_eval_window=True,
                )
                if rec is None:
                    return
                sepa.setup_id = rec["setup_id"]
                sepa.original_base_start = rec.get("original_base_start")
                sepa.left_censored = bool(rec.get("left_censored"))
                sepa.lifecycle_status = str(rec.get("status") or "")
                sid = rec["setup_id"]
                if sepa.vcp_detected:
                    _unique("vcp_detected", sid)
                if sepa.pivot:
                    _unique("valid_pivot", sid)
                if sepa.entry_valid:
                    _unique("entry_ready", sid)
                if sepa.left_censored:
                    _unique("left_censored", sid)
                    return
                if rec.get("status") == "PIVOT_RETEST":
                    _unique("pivot_retest", sid)
                    fill_counts[variant]["PIVOT_RETEST"] += 1
                    return
                if not ledger.is_core_opportunity(sym):
                    return
                if sid not in seen_unique["setups"]:
                    seen_unique["setups"].add(sid)
                    setups.append({
                        "setup_id": sid,
                        "symbol": sym,
                        "detection_date": as_of,
                        "original_base_start": rec.get("original_base_start"),
                        "stage2": sepa.structure_pass,
                        "rs": sepa.rs_percentile,
                        "vcp_state": sepa.vcp_state,
                        "pivot": sepa.pivot,
                        "buy_zone_low": sepa.buy_zone_low,
                        "buy_zone_high": sepa.buy_zone_high,
                        "structural_stop": sepa.structural_stop,
                        "left_censored": False,
                        "experiment_version": cfg.eligibility_version,
                        "variant": variant,
                    })
                if e_opportunity is False:
                    return
                if sepa.extended:
                    _unique("observed_extended", sid)
                    fill_counts[variant][FILL_EXTENDED] += 1
                    if rec.get("saw_entry_ready"):
                        ledger.mark(sym, "EXTENDED")
                    return
                if sepa.entry_rejection == "WIDE_STRUCTURAL_STOP":
                    _unique("stop_too_wide", sid)
                    return
                if not sepa.entry_valid or not sepa.stop_ok or fwd is None:
                    return
                packed = sepa_fill_sim(
                    fwd, stop=float(sepa.structural_stop), pivot=sepa.pivot,
                    buy_zone_low=sepa.buy_zone_low, buy_zone_high=sepa.buy_zone_high,
                    horizon=horizon,
                )
                cls = packed["class"]
                if packed.get("sim") is None:
                    fill_counts[variant][cls] += 1
                    if cls == FILL_GAP_THROUGH:
                        _unique("gap_through", sid)
                    elif cls in {"FAILED", "EXPIRED", "INVALIDATED", "MISSED"}:
                        _unique("expired_failed", sid)
                    ledger.mark(sym, cls)
                    return
                sim = attach_session_path(dict(packed["sim"]), fwd, as_of=as_of)
                exit_s = str(sim.get("exit_date") or as_of)
                if timeline.horizon_crosses(sym, as_of, exit_s):
                    fill_counts[variant][FILL_CA_CENSORED] += 1
                    diagnostics["ca_censored_outcomes"] += 1
                    _unique("ca_censored", sid)
                    ledger.mark(sym, FILL_CA_CENSORED)
                    return
                _unique("valid_fill", sid)
                _record(variant, sim, sepa, as_of, cls, fwd, snap.membership_hash, snap.source)
                ledger.mark(sym, "FILLED")

            if "E" in variants and observe_lifecycle_daily():
                e_opp = attempt_e_entry(
                    scanner_ok=scanner_ok,
                    structure_pass=sepa.structure_pass,
                    rs_pass=sepa.rs_pass,
                    vcp_detected=sepa.vcp_detected,
                )
                if e_opp and sepa.entry_valid and step5_skip:
                    diagnostics["scanner_step5_missed_e_entry_ready"] += 1
                _sepa_lifecycle("E", e_opportunity=e_opp)
            if "F" in variants and sepa.trend_template_pass and sepa.vcp_detected:
                _sepa_lifecycle("F", e_opportunity=None)

        if di % 25 == 0:
            print(
                f"SEPA-001R2 {as_of} {di+1}/{len(eval_dates)} "
                f"investable={len(snap.investable)} stage2={s2} rs={rs_n} vcp={vcp_n} "
                f"F_n={len(rows.get('F') or [])}",
                flush=True,
            )

    n_years = max(0.01, n_as_of / 252.0)
    n_trials = max(7, len(variants))
    summary: dict[str, Any] = {}

    def _by_block(seq, key="as_of"):
        out = {"development": [], "validation": [], "confirmation": [], "other": []}
        for item in seq:
            d = item[key] if isinstance(item, dict) else getattr(item, key)
            out[block_of(str(d))].append(item)
        return out

    for v in variants:
        if v == "G":
            stats = summarize_signal_study(g_rows)
            stats["n_raw_signal_days"] = len(g_rows_raw)
            stats["n_deduped"] = len(g_rows)
            stats["statistical_unit"] = (
                "raw=Stage-2+RS signal-days; deduped=symbol embargoed until actual exit session"
            )
            stats["fill_attempt_counts"] = dict(fill_counts[v])
            stats["layer"] = "signal"
            stats["deployment"] = {
                "deployment_eligible": False,
                "paper_shadow": False,
                "label": "NOT_DEPLOYMENT_ELIGIBLE",
                "reasons": ["G is a pure signal study, not core SEPA"],
            }
            stats["statistical_verdict"] = "NOT_SEPA_R"
            blocks = _by_block(g_rows)
            stats["walk_forward"] = {
                name: summarize_signal_study(items) for name, items in blocks.items()
            }
            summary[v] = stats
            continue
        deduped = rows[v]
        raw = rows_raw[v]
        stats = summarize_r(deduped, n_years=n_years, n_trials=n_trials)
        gate = statistical_gate([x.net_r for x in deduped], n_trials=n_trials)
        ci = gate.get("block_ci") or stats.get("block_ci") or {}
        lo = ci.get("ci_lower") if isinstance(ci, dict) else None
        wf = _by_block(deduped)
        conf = wf["confirmation"]
        unseen_n = len(conf)
        has_unseen = unseen_n > 0
        dep = deployment_eligible(
            statistical=gate,
            pit_class=str((integ or {}).get("overall") or "PIT_UNVERIFIED"),
            ca_complete=bool(ca.get("ca_complete")),
            ca_research_acceptable=bool((integ or {}).get("ca_research_acceptable")),
            n_post_warmup_years=n_years,
            has_unseen_block=has_unseen if v == "F" else False,
            unseen_n=unseen_n if v == "F" else 0,
            ci_lower_ok=bool(lo is not None and float(lo) >= 0),
            known_lookahead=False,
            causality_ok=True,
        )
        stats["statistical_verdict"] = gate.get("statistical_verdict")
        stats["deployment"] = dep
        stats["fill_attempt_counts"] = dict(fill_counts[v])
        stats["layer"] = "signal" if v in ("A", "B", "C", "D") else "setup_entry"
        stats["n_raw_signal_days"] = len(raw)
        stats["n_deduped"] = len(deduped)
        stats["statistical_unit"] = (
            "raw=daily scanner rows; deduped=symbol embargoed until actual exit session"
            if v in ("A", "B", "C", "D") else
            "persistent setup identity; core F/E fills only"
        )
        stats["walk_forward"] = {
            name: summarize_r(items, n_years=max(0.01, len(items) / 50.0), n_trials=n_trials)
            for name, items in wf.items()
        }
        for name, items in wf.items():
            g2 = statistical_gate([x.net_r for x in items], n_trials=n_trials)
            stats["walk_forward"][name]["statistical_verdict"] = g2.get("statistical_verdict")
            stats["walk_forward"][name]["n"] = len(items)
        summary[v] = stats

    def _bucket_pack(xs):
        arr = np.array(xs, dtype=float)
        return {
            "n": int(arr.size),
            "mean_fwd_20d": None if arr.size == 0 else round(float(arr.mean()) * 100.0, 3),
            "median_fwd_20d": None if arr.size == 0 else round(float(np.median(arr)) * 100.0, 3),
        }

    yearly_out = {}
    for y, rec in yearly_universe.items():
        n = max(1, rec["as_of_points"])
        yearly_out[y] = {
            "as_of_points": rec["as_of_points"],
            "mean_candidates": round(rec["candidates_sum"] / n, 1),
            "mean_investable": round(rec["investable_sum"] / n, 1),
            "exclusions": dict(rec["exclusions"]),
        }

    return {
        "experiment": "SEPA-001R2",
        "revision": "SEPA-001R2.1",
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config_hash": cfg.config_hash(),
        "eligibility_version": cfg.eligibility_version,
        "pivot_version": cfg.pivot_version,
        "vcp_version": cfg.vcp_version,
        "layer": "signal_quality_primary",
        "validation_protocol": {
            "development_end": DEV_END,
            "validation": [VAL_START, VAL_END],
            "confirmation": [CONF_START, CONF_END],
            "declared_before_results": True,
        },
        "sample": {
            "symbols_loaded": len(frames),
            "n_as_of": n_as_of,
            "warmup_sessions": warmup_sessions,
            "evaluation_start": evaluation_start,
            "evaluation_end": evaluation_end,
            "horizon": horizon,
            "date_step": date_step,
            "scanner_step": scanner_step,
            "top_n": top_n,
            "unique_setups": len(seen_unique.get("setups") or []),
            "left_censored_unique": len(seen_unique.get("left_censored") or []),
        },
        "pit": {
            "universe_source": "bhav_inferred",
            "universe_complete": False,
            "research_grade": False,
            "ca_complete": ca.get("ca_complete"),
            "as_of_metadata": True,
            "last_universe": last_snap_meta,
            "note": "Each decision carries as-of bhav_inferred membership; not reused from the final date.",
        },
        "integrity": integ,
        "funnel_snapshots": funnel_snap,
        "funnel_unique": {k: funnel_unique.get(k, 0) for k in UNIQUE_FUNNEL_KEYS},
        "yearly_universe": yearly_out,
        "variants": summary,
        "setups": setups,
        "rs_buckets": {k: _bucket_pack(v) for k, v in rs_bucket_fwd.items()},
        "quarantine_n_static_catalogue": len(static_q),
        "ca_events_n": len(timeline.rows),
        "rs_threshold": cfg.rs_threshold,
        "diagnostics": diagnostics,
        "g_rows": g_rows,
    }


def persist_r2(payload: dict[str, Any], name: str = "ablation_001r2.json") -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    slim = dict(payload)
    setups = slim.pop("setups", [])
    g_rows = slim.pop("g_rows", [])
    path.write_text(json.dumps(slim, indent=2, default=str))
    side = _OUT / "setups.jsonl"
    with side.open("w", encoding="utf-8") as fh:
        for row in setups:
            fh.write(json.dumps(row, default=str) + "\n")
    g_path = _OUT / "g_signal_rows.jsonl"
    with g_path.open("w", encoding="utf-8") as fh:
        for row in g_rows:
            fh.write(json.dumps(row, default=str) + "\n")
    opp = _OUT / "opportunities.jsonl"
    with opp.open("w", encoding="utf-8") as fh:
        for row in setups:
            fh.write(json.dumps({"kind": "setup", **row}, default=str) + "\n")
        for vid, stats in (payload.get("variants") or {}).items():
            if vid == "G":
                continue
            n = stats.get("n_deduped")
            fh.write(json.dumps({"kind": "variant_summary", "variant": vid, "n_deduped": n}, default=str) + "\n")
    return path
