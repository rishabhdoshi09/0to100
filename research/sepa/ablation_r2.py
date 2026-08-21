"""SEPA-001R2 ablation — as-of universe, lifecycle, funnel. Research only."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

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
from research.sepa.ca_audit import quarantine_symbols, unresolved_events, verify_report
from research.sepa.config import R2_CONFIG, SepaConfig
from research.sepa.engine import evaluate_sepa_eligibility
from research.sepa.entry import FILL_EXTENDED, FILL_GAP_THROUGH
from research.sepa.frames import ca_status, iso_date, pit_universe, slice_as_of
from research.sepa.gates import deployment_eligible, statistical_gate
from research.sepa.integrity import research_integrity_report
from research.sepa.lifecycle import PersistentSetupLedger
from research.sepa.rs import build_rs_table
from research.sepa.types import SepaEligibility
from research.sepa.universe_pit import FastInvestable, screen_investable_as_of

_OUT = Path(__file__).resolve().parents[2] / "docs" / "overhaul" / "experiments" / "SEPA-001R2"

FUNNEL_KEYS = (
    "candidates",
    "investable",
    "stage2",
    "rs_pass",
    "vcp_forming",
    "vcp_confirmed",
    "pivot_defined",
    "entry_ready",
    "gap_through",
    "extended_missed",
    "left_censored",
    "stop_too_wide",
    "valid_fill",
    "pivot_retest",
)


def _session_calendar(frames: Mapping[str, pd.DataFrame]) -> list[pd.Timestamp]:
    acc = set()
    for df in frames.values():
        if df is None or len(df) == 0:
            continue
        acc.update(pd.DatetimeIndex(df.index).tz_localize(None).normalize())
    return sorted(acc)


def _fwd_after(df: pd.DataFrame, as_of, horizon: int) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return None
    hist = slice_as_of(df, as_of)
    if hist is None or len(hist) == 0:
        return None
    rest = df[pd.DatetimeIndex(df.index).normalize() > pd.Timestamp(iso_date(as_of))]
    if rest is None or len(rest) == 0:
        return None
    return rest.iloc[: int(horizon)]


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
    scanner_step: int = 5,
    quarantined: set[str] | None = None,
    integrity: dict[str, Any] | None = None,
    rs_threshold: float | None = None,
) -> dict[str, Any]:
    """Date-major PIT runner. Layer 1 = unique-setup signal quality."""
    cfg = config or R2_CONFIG
    if rs_threshold is not None:
        cfg = replace(cfg, rs_threshold=float(rs_threshold))
    qset = set(quarantined or set())
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
    fast = FastInvestable(frames)
    versions = {
        "eligibility_version": cfg.eligibility_version,
        "vcp_version": cfg.vcp_version,
        "pivot_version": cfg.pivot_version,
    }
    ledgers = {v: PersistentSetupLedger(versions=versions) for v in variants if v in ("E", "F")}
    need_scanner = any(v in variants for v in ("A", "B", "C", "D", "E"))
    scanner = None
    if need_scanner:
        from scan.unified_scanner import UnifiedScanner
        scanner = UnifiedScanner()

    regimes = _nifty_regimes()
    rows: dict[str, list[SimRowR]] = {v: [] for v in variants}
    last_exit = {v: {} for v in variants}
    fill_counts = {v: defaultdict(int) for v in variants}
    funnel = {k: 0 for k in FUNNEL_KEYS}
    funnel_unique = {k: 0 for k in FUNNEL_KEYS}
    seen_unique: dict[str, set[str]] = defaultdict(set)
    yearly_universe: dict[str, dict[str, int]] = {}
    setups: list[dict[str, Any]] = []
    rs_bucket_fwd: dict[str, list[float]] = defaultdict(list)
    n_as_of = 0
    last_snap_meta = None

    def _record(variant: str, sim: dict[str, Any] | None, sepa: SepaEligibility,
                as_of: str, t_key: str, fill_class: str, scan_hit=None):
        if sim is None:
            fill_counts[variant][fill_class] += 1
            return
        if t_key <= last_exit[variant].get(sepa.symbol, ""):
            return
        hold = int(sim["hold"])
        try:
            exit_s = str((pd.Timestamp(as_of) + pd.Timedelta(days=max(1, hold))).date())
        except Exception:
            exit_s = as_of
        last_exit[variant][sepa.symbol] = exit_s
        scanner_fill = variant in ("A", "B", "C", "D")
        if scanner_fill and scan_hit is not None:
            stop_px = float(scan_hit.stop)
            net = _net(sim["gross_r"], sim["entry"], stop_px)
        else:
            stop_px = float(sepa.structural_stop or sim.get("entry", 1) * 0.95)
            net = _net(sim["gross_r"], sim["entry"], stop_px)
        ext = None
        if sepa.pivot:
            ext = (float(sim["entry"]) / float(sepa.pivot) - 1.0) * 100.0
        fill_counts[variant][fill_class] += 1
        rows[variant].append(SimRowR(
            variant=variant, symbol=sepa.symbol, as_of=as_of,
            entry=float(sim["entry"]), stop=stop_px,
            net_r=float(net), gross_r=float(sim["gross_r"]),
            hold=hold, outcome=str(sim["outcome"]),
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
        ))

    pit_u = pit_universe(eval_dates[-1])
    ca = ca_status()
    integ = integrity if integrity is not None else research_integrity_report(
        frames=frames, as_of=eval_dates[-1],
    )

    for di, as_ts in enumerate(eval_dates):
        as_of = iso_date(as_ts)
        n_as_of += 1
        snap = fast.snapshot(
            as_of, min_price=min_price, min_turnover=min_turnover,
            min_sessions=min_sessions, quarantined=qset, top_n=top_n,
            source=str(pit_u.get("source") or "bhav_inferred"),
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

        funnel["candidates"] += len(snap.candidates)
        funnel["investable"] += len(snap.investable)
        rs_table = build_rs_table(frames, as_of, cfg, universe=snap.investable)
        run_scanner_today = scanner is not None and (di % max(1, int(scanner_step)) == 0)

        for sym in snap.investable:
            df = fast.frame(sym)
            hist = slice_as_of(df, as_of)
            if hist is None or len(hist) < min(80, min_sessions):
                continue
            fwd = _fwd_after(df, as_of, horizon)
            meta = {
                "universe_complete": bool(pit_u.get("universe_complete")),
                "universe_source": snap.source,
                "universe_note": pit_u.get("note") or "",
                "research_grade": bool(pit_u.get("research_grade")),
                "ca_complete": bool(ca.get("ca_complete")),
                "ca_note": ca.get("note") or "",
                "ca_verified": bool(ca.get("verified")),
                "ca_n_events": int(ca.get("n_events") or 0),
                **snap.to_meta(sym),
            }
            sepa = evaluate_sepa_eligibility(
                sym, as_of, frame=hist, frames=frames, universe=snap.investable,
                rs_table=rs_table, config=cfg, pit_meta=meta,
            )
            if sepa.rs_percentile is not None and fwd is not None and len(fwd) >= horizon:
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
                funnel["stage2"] += 1
            if sepa.structure_pass and sepa.rs_pass:
                funnel["rs_pass"] += 1
            if sepa.vcp_state in {"BASE_FORMING", "CONTRACTION_1", "CONTRACTION_2", "VCP_FORMING"}:
                funnel["vcp_forming"] += 1
            if sepa.vcp_detected:
                funnel["vcp_confirmed"] += 1
            if sepa.vcp_state in {"PIVOT_DEFINED", "ENTRY_READY"} or sepa.pivot:
                funnel["pivot_defined"] += 1
            if sepa.entry_valid:
                funnel["entry_ready"] += 1
            if sepa.entry_rejection == "WIDE_STRUCTURAL_STOP":
                funnel["stop_too_wide"] += 1

            scan_hit = None
            scanner_ok = False
            if run_scanner_today and scanner is not None:
                try:
                    scan_hit = scanner._analyze(sym, hist)
                except Exception:
                    scan_hit = None
                scanner_ok = bool(scan_hit is not None and getattr(scan_hit, "signals", None))

            if "A" in variants and scanner_ok and scan_hit is not None and fwd is not None:
                _record("A", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                          float(scan_hit.target), horizon),
                        sepa, as_of, as_of, "SCANNER", scan_hit)
            if "B" in variants and scanner_ok and sepa.structure_pass and scan_hit is not None and fwd is not None:
                _record("B", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                          float(scan_hit.target), horizon),
                        sepa, as_of, as_of, "SCANNER", scan_hit)
            if "C" in variants and scanner_ok and sepa.structure_pass and sepa.rs_pass and scan_hit is not None and fwd is not None:
                _record("C", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                          float(scan_hit.target), horizon),
                        sepa, as_of, as_of, "SCANNER", scan_hit)
            if "D" in variants and scanner_ok and sepa.structure_pass and sepa.rs_pass and sepa.vcp_detected and scan_hit is not None and fwd is not None:
                _record("D", _scanner_sim(fwd, float(scan_hit.entry), float(scan_hit.stop),
                                          float(scan_hit.target), horizon),
                        sepa, as_of, as_of, "SCANNER", scan_hit)

            if "G" in variants and sepa.structure_pass and sepa.rs_pass and fwd is not None and len(fwd) >= 1:
                # Signal layer: next-open vs last close, 20d R using 1% risk proxy is
                # not SEPA. Record forward % converted to a 1-R proxy of 8% stop.
                o = float(fwd["open"].iloc[0])
                last = float(fwd["close"].iloc[min(horizon, len(fwd)) - 1])
                stop_px = o * 0.92
                risk = o - stop_px
                if risk > 0 and o > 0:
                    gross = (last - o) / risk
                    _record("G", {
                        "entry": o, "gross_r": gross, "hold": min(horizon, len(fwd)),
                        "outcome": "WIN" if gross > 0 else "LOSS",
                        "mae_r": 0.0, "mfe_r": max(0.0, gross),
                        "reached_1r": gross >= 1, "reached_2r": gross >= 2,
                        "stop_before_1r": False, "failed_break": False,
                    }, sepa, as_of, as_of, "SIGNAL_OPEN")

            def _sepa_once(variant: str, ok: bool):
                if variant not in ledgers or not ok or not sepa.pivot:
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
                if sepa.left_censored:
                    funnel["left_censored"] += 1
                    if sid not in seen_unique["left_censored"]:
                        seen_unique["left_censored"].add(sid)
                        funnel_unique["left_censored"] += 1
                    return
                if rec.get("status") == "PIVOT_RETEST":
                    funnel["pivot_retest"] += 1
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
                    })
                if sepa.extended:
                    funnel["extended_missed"] += 1
                    fill_counts[variant][FILL_EXTENDED] += 1
                    if rec.get("saw_entry_ready"):
                        ledger.mark(sym, "EXTENDED")
                    return
                if sepa.entry_rejection == "WIDE_STRUCTURAL_STOP":
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
                        funnel["gap_through"] += 1
                    ledger.mark(sym, cls)
                    return
                funnel["valid_fill"] += 1
                _record(variant, packed["sim"], sepa, as_of, as_of, cls)
                ledger.mark(sym, "FILLED")

            if "E" in variants:
                _sepa_once("E", run_scanner_today and scanner_ok and sepa.structure_pass
                           and sepa.rs_pass and sepa.vcp_detected)
            if "F" in variants:
                _sepa_once("F", sepa.trend_template_pass and sepa.vcp_detected)

    n_years = max(0.01, n_as_of / 252.0)
    n_trials = max(7, len(variants))
    summary = {}
    for v in variants:
        stats = summarize_r(rows[v], n_years=n_years, n_trials=n_trials)
        gate = statistical_gate([x.net_r for x in rows[v]], n_trials=n_trials)
        ci = gate.get("block_ci") or {}
        lo = ci.get("ci_lower") if isinstance(ci, dict) else None
        dep = deployment_eligible(
            statistical=gate,
            pit_class=str((integ or {}).get("overall") or "PIT_UNVERIFIED"),
            ca_complete=bool(ca.get("ca_complete")),
            n_post_warmup_years=n_years,
            has_unseen_block=False,
            unseen_n=0,
            ci_lower_ok=bool(lo is not None and float(lo) >= 0),
            known_lookahead=False,
            causality_ok=True,
        )
        stats["statistical_verdict"] = gate.get("statistical_verdict")
        stats["deployment"] = dep
        stats["fill_attempt_counts"] = dict(fill_counts[v])
        stats["layer"] = "signal" if v in ("A", "B", "C", "D", "G") else "setup_entry"
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
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config_hash": cfg.config_hash(),
        "eligibility_version": cfg.eligibility_version,
        "pivot_version": cfg.pivot_version,
        "vcp_version": cfg.vcp_version,
        "layer": "signal_quality_primary",
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
            "universe_source": pit_u.get("source"),
            "universe_complete": pit_u.get("universe_complete"),
            "research_grade": pit_u.get("research_grade"),
            "ca_complete": ca.get("ca_complete"),
            "last_universe": last_snap_meta,
        },
        "integrity": integ,
        "funnel_snapshots": funnel,
        "funnel_unique": {k: len(v) if isinstance(v, set) else v for k, v in {
            **funnel_unique, "setups": seen_unique.get("setups") or set(),
        }.items()},
        "yearly_universe": yearly_out,
        "variants": summary,
        "setups": setups,
        "rs_buckets": {k: _bucket_pack(v) for k, v in rs_bucket_fwd.items()},
        "quarantine_n": len(qset),
        "rs_threshold": cfg.rs_threshold,
    }


def persist_r2(payload: dict[str, Any], name: str = "ablation_001r2.json") -> Path:
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / name
    slim = dict(payload)
    setups = slim.pop("setups", [])
    path.write_text(json.dumps(slim, indent=2, default=str))
    side = _OUT / "setups.jsonl"
    with side.open("w", encoding="utf-8") as fh:
        for row in setups:
            fh.write(json.dumps(row, default=str) + "\n")
    return path
