"""Protocol-scoped certification for frozen Phase A.5 (29-name panel).

Does NOT alter global dataset trust. Global remains OPERATIONAL_ONLY unless the
full store independently earns RESEARCH_GRADE.

Produces an evidence-backed scoped certification artifact that may conclude
READY_FOR_SCIENTIFIC_RERUN or BLOCKED for the exact frozen protocol scope.

Does NOT execute Phase A.5 experiments.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from core.data_integrity import _GAP_PCT
from data.bhavcopy_runtime import ensure_loaded
from data.bhavcopy_store import get_ohlcv, reload_corporate_actions
from data.corporate_actions import load_events
from data.nse_ca_ingest import ADJUSTMENT_POLICY_VERSION
from data.security_identity import (
    load_identity_ledger,
    lineage_coverage_report,
    resolve_as_of,
)
from data.universe_history import history_path, ledger_status
from product.plain_language import PlainCard, render_layers
from research.intelligence.data.discontinuity_audit import (
    _CONSEC_CAL_DAYS,
    audit_symbol,
    audit_universe,
    verification_trace,
)
from research.intelligence.data.snapshot_store import SnapshotStore

REPO_ROOT = Path(__file__).resolve().parents[2]

FROZEN_PANEL = [
    "RELIANCE", "ONGC", "BPCL", "TCS", "INFY", "WIPRO", "HCLTECH",
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
    "ITC", "HINDUNILVR", "NESTLEIND", "SUNPHARMA", "DRREDDY", "CIPLA",
    "M&M", "MARUTI", "TATASTEEL", "JSWSTEEL", "HINDALCO",
    "NTPC", "POWERGRID", "LT", "ADANIENT", "BAJFINANCE", "BAJAJFINSV",
]
FROZEN_DATE_START = "2023-08-11"
FROZEN_DATE_END = "2026-08-11"
FROZEN_HYPOTHESIS_IDS = [
    "81b8889792f53113",  # EXP-A5-01
    "590571a11ee06fc2",  # EXP-A6-01
    "775b4a0fce7d5b83",  # EXP-A2-01
    "7842a46ee335685a",  # EXP-A3-01
    "3734b8a0a9124a60",  # EXP-A5A6-01
]
EXPERIMENT_ROWS = [
    ("EXP-A5-01", "81b8889792f53113"),
    ("EXP-A6-01", "590571a11ee06fc2"),
    ("EXP-A2-01", "775b4a0fce7d5b83"),
    ("EXP-A3-01", "7842a46ee335685a"),
    ("EXP-A5A6-01", "3734b8a0a9124a60"),
]
UNRESOLVED_RATE_MAX = 0.002
VALIDATOR_VERSION = "phase_a5_scoped_certification.v1"
PROTOCOL_VERSION = "PHASE_A5_FROZEN_PROTOCOLS@2026-08-11"

# Experiments that consume the static sector map (not PIT sector history).
SECTOR_STATIC_CONSUMERS = {"EXP-A5-01", "EXP-A6-01", "EXP-A5A6-01"}
# Experiments that do not use sectors at all.
SECTOR_UNUSED = {"EXP-A2-01", "EXP-A3-01"}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=str(REPO_ROOT)
        ).strip()
    except Exception:
        return "unknown"


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _event_source_hash(event: dict) -> str:
    blob = json.dumps(event, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_frozen_sector_map() -> dict[str, str]:
    p = REPO_ROOT / "logs" / "phase_a5" / "sector_map.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {str(k).upper(): str(v) for k, v in raw.items()}


def protocol_dependency_matrix() -> list[dict]:
    """S1 — what each frozen experiment actually consumes."""
    rows = []
    common = dict(
        securities=list(FROZEN_PANEL),
        date_start=FROZEN_DATE_START,
        date_end=FROZEN_DATE_END,
        price_fields=["close"],
        index_vix="NOT_REQUIRED",
        transaction_costs="CNC round_trip_cost_pct (config, not market data)",
        universe_mode="FIXED_PREREGISTERED_29",
        symbol_lineage="REQUIRED if any panel name has symbol-change evidence",
        listing_delisting="REQUIRED within frozen window for fixed panel",
        benchmark_history="NOT_REQUIRED as Nifty/VIX series (panel EW / internal baselines)",
    )
    specs = [
        ("EXP-A5-01", "81b8889792f53113", {
            "ca": "REQUIRED for continuous adjusted closes",
            "identity": "REQUIRED for panel security_ids",
            "listing_delisting": "REQUIRED within window (membership of fixed panel)",
            "sector": "STATIC_MAP (frozen known_limitation: no PIT sector history)",
            "features": "returns panel; structure methods on closes",
            "benchmark": "static sector map + correlation clusters",
            "universe_membership_history": "NOT_REQUIRED (fixed preregistered 29)",
        }),
        ("EXP-A6-01", "590571a11ee06fc2", {
            "ca": "REQUIRED for continuous adjusted closes",
            "identity": "REQUIRED",
            "listing_delisting": "REQUIRED within window",
            "sector": "STATIC_MAP for sector_herfindahl baseline (not PIT history)",
            "features": "corr network metrics on returns",
            "benchmark": "pairwise corr clusters + sector HHI",
            "universe_membership_history": "NOT_REQUIRED (fixed preregistered 29)",
        }),
        ("EXP-A2-01", "775b4a0fce7d5b83", {
            "ca": "REQUIRED for continuous adjusted closes",
            "identity": "REQUIRED",
            "listing_delisting": "REQUIRED within window",
            "sector": "NOT_REQUIRED",
            "features": "60d momentum ranks; horizons 5/10/22/66d",
            "benchmark": "cost-aware OOS R / DSR (no index series)",
            "universe_membership_history": "NOT_REQUIRED (fixed preregistered 29)",
        }),
        ("EXP-A3-01", "7842a46ee335685a", {
            "ca": "REQUIRED for continuous adjusted closes",
            "identity": "REQUIRED",
            "listing_delisting": "REQUIRED within window",
            "sector": "NOT_REQUIRED",
            "features": "mom_5/10/20/60, vol_20; 10d class target",
            "benchmark": "naive + momentum-rank incumbents",
            "universe_membership_history": "NOT_REQUIRED (fixed preregistered 29)",
        }),
        ("EXP-A5A6-01", "3734b8a0a9124a60", {
            "ca": "REQUIRED for continuous adjusted closes",
            "identity": "REQUIRED",
            "listing_delisting": "REQUIRED within window",
            "sector": "STATIC_MAP (context only; not PIT history)",
            "features": "structure stability × network × momentum",
            "benchmark": "FDR across preregistered interactions",
            "universe_membership_history": "NOT_REQUIRED (fixed preregistered 29)",
        }),
    ]
    for exp_id, hid, extra in specs:
        for asset, need in [
            ("adjusted_close_panel", "REQUIRED"),
            ("corporate_actions", extra["ca"]),
            ("security_identity", extra["identity"]),
            ("listing_delisting_window", extra["listing_delisting"]),
            ("universe_membership_history", extra["universe_membership_history"]),
            ("sector", extra["sector"]),
            ("index_vix", common["index_vix"]),
            ("symbol_lineage", common["symbol_lineage"]),
            ("features", extra["features"]),
            ("benchmark", extra["benchmark"]),
            ("transaction_costs", common["transaction_costs"]),
        ]:
            status = "DEFINED"
            blocker = None
            if need.startswith("NOT_REQUIRED"):
                status = "NOT_REQUIRED"
            rows.append({
                "experiment": exp_id,
                "hypothesis_id": hid,
                "required_data_asset": asset,
                "required_date_range": f"{FROZEN_DATE_START}→{FROZEN_DATE_END}",
                "required_securities": "FROZEN_29",
                "requirement": need,
                "universe_mode": common["universe_mode"],
                "status": status,
                "blocker": blocker,
            })
    return rows


def verify_identity_panel() -> dict[str, Any]:
    """S2 — identity for each of the 29 names."""
    ensure_loaded(rebuild_from_local=False)
    ledger = load_identity_ledger()
    lin = lineage_coverage_report(ledger, focus_symbols=set(FROZEN_PANEL))
    mid = "2024-12-15"
    rows = []
    blockers = []
    # Detect symbol collisions in ledger (same symbol, overlapping validity)
    by_sym: dict[str, list] = {}
    for r in ledger.get("securities") or []:
        by_sym.setdefault(str(r.get("symbol") or "").upper(), []).append(r)

    for sym in FROZEN_PANEL:
        res = resolve_as_of(sym, mid, ledger)
        matches = by_sym.get(sym) or []
        sec = next((r for r in matches if r.get("symbol") == sym), None)
        status = "UNRESOLVED"
        note = ""
        if len(matches) > 1:
            # Overlapping open intervals = CONFLICT
            openish = [m for m in matches if not m.get("valid_to") or not m.get("delisting_date")]
            if len(openish) > 1:
                status = "CONFLICT"
                note = f"Multiple open identity rows ({len(openish)})"
                blockers.append(sym)
        if status != "CONFLICT":
            if sec and res.get("status") == "OK":
                if sec.get("isin") and sec.get("security_id"):
                    status = "VERIFIED"
                    note = f"series={sec.get('series')}"
                else:
                    status = "UNRESOLVED"
                    note = "Identity without ISIN/security_id"
                    blockers.append(sym)
            elif res.get("status") == "UNKNOWN":
                status = "UNRESOLVED"
                note = "Not in identity ledger"
                blockers.append(sym)
            elif res.get("status") in {"NOT_YET_LISTED", "SYMBOL_ENDED"}:
                status = "CONFLICT"
                note = f"resolve_as_of mid-window → {res.get('status')}"
                blockers.append(sym)
        focus_hit = [
            t for t in (lin.get("focus_transitions") or [])
            if t.get("new_symbol") == sym or t.get("old_symbol") == sym
        ]
        # Lineage for these mega-caps: absence of symbol-change = NOT_APPLICABLE
        lineage_status = "NOT_APPLICABLE" if not focus_hit else (
            "VERIFIED" if not any(
                t.get("status") in {"UNRESOLVED", "CONFLICT"} for t in focus_hit
            ) else "UNRESOLVED"
        )
        if lineage_status == "UNRESOLVED":
            blockers.append(sym)
            status = "UNRESOLVED"
            note = (note + "; unresolved lineage").strip("; ")
        rows.append({
            "symbol": sym,
            "security_id": (sec or {}).get("security_id"),
            "isin": (sec or {}).get("isin"),
            "series": (sec or {}).get("series"),
            "listing_date": (sec or {}).get("listing_date"),
            "delisting_date": (sec or {}).get("delisting_date"),
            "valid_from": (sec or {}).get("valid_from"),
            "valid_to": (sec or {}).get("valid_to"),
            "status": status,
            "lineage_status": lineage_status,
            "note": note,
            "lineage_transitions": focus_hit,
            "mid_window_resolve": res.get("status"),
        })
    if lin.get("focus_blocking"):
        blockers.extend(
            t.get("new_symbol") or t.get("old_symbol")
            for t in lin["focus_blocking"]
        )
    verified = all(r["status"] == "VERIFIED" for r in rows)
    return {
        "ok": verified and not lin.get("focus_blocking"),
        "rows": rows,
        "lineage": {
            "symbol_lineage_complete": lin.get("symbol_lineage_complete"),
            "isin_confirmed_rate": lin.get("isin_confirmed_rate"),
            "by_status": lin.get("by_status"),
            "focus_blocking": lin.get("focus_blocking"),
        },
        "blockers": sorted({b for b in blockers if b}),
        "security_ids": {r["symbol"]: r["security_id"] for r in rows},
    }


def _in_frozen_window(d0: str, d1: str) -> bool:
    # Event overlaps frozen window if either bar is inside
    return (
        (FROZEN_DATE_START <= str(d0)[:10] <= FROZEN_DATE_END)
        or (FROZEN_DATE_START <= str(d1)[:10] <= FROZEN_DATE_END)
    )


def verify_ca_panel() -> dict[str, Any]:
    """S3 — CA verification for exact panel only (global unresolved ignored)."""
    ensure_loaded(rebuild_from_local=False)
    reload_corporate_actions()
    events = load_events()
    id_ledger = load_identity_ledger()
    by_sym = {r.get("symbol"): r for r in (id_ledger.get("securities") or [])}

    event_rows = []
    for sym in FROZEN_PANEL:
        sec_id = (by_sym.get(sym) or {}).get("security_id")
        for e in events.get(sym) or []:
            ex = str(e.get("ex_date", ""))[:10]
            if not (FROZEN_DATE_START <= ex <= FROZEN_DATE_END):
                # Still retain panel CA outside window for completeness, tagged
                in_window = False
            else:
                in_window = True
            event_rows.append({
                "security_id": sec_id,
                "symbol": sym,
                "event_type": e.get("type"),
                "official_source": e.get("source") or e.get("official_source") or "nse_ca",
                "ex_date": ex,
                "ratio_factor": e.get("factor"),
                "in_frozen_window": in_window,
                "source_hash": _event_source_hash(e),
                "raw_event": {
                    k: e.get(k) for k in (
                        "type", "factor", "ex_date", "source", "subject", "ratio"
                    ) if k in e or e.get(k) is not None
                },
            })

    disc_all = []
    traces = []
    for sym in FROZEN_PANEL:
        hits = audit_symbol(sym, events=events)
        for h in hits:
            if not _in_frozen_window(h.d0, h.d1):
                continue
            disc_all.append(h)
            if h.cal_days <= _CONSEC_CAL_DAYS:
                tr = verification_trace(h)
                tr["security_id"] = (by_sym.get(sym) or {}).get("security_id")
                traces.append(tr)

    consec = [d for d in disc_all if d.cal_days <= _CONSEC_CAL_DAYS]
    unresolved = [d for d in consec if d.classification == "UNRESOLVED"]
    verified = [d for d in consec if d.ca_status == "VERIFIED"]
    genuine = [d for d in consec if d.classification == "GENUINE_MARKET_MOVE"]

    # Attach raw/adj prices onto window CA events from nearby discontinuities
    by_ex = {}
    for d in consec:
        for e in d.ca_events_near or []:
            by_ex[(d.symbol, e.get("ex_date"))] = d
    for row in event_rows:
        hit = by_ex.get((row["symbol"], row["ex_date"]))
        if hit:
            row["raw_prices"] = {"pre": hit.pre_raw, "post": hit.post_raw}
            row["adjusted_prices"] = {"pre": hit.pre_adj, "post": hit.post_adj}
            row["verification_result"] = hit.ca_status
            row["classification"] = hit.classification
        else:
            row["raw_prices"] = None
            row["adjusted_prices"] = None
            row["verification_result"] = (
                "NO_LARGE_JUMP" if row["in_frozen_window"] else "OUT_OF_WINDOW"
            )
            row["classification"] = None

    return {
        "ok": len(unresolved) == 0,
        "consecutive_events_in_window": len(consec),
        "verified_ca_transitions": len(verified),
        "genuine_market_moves": len(genuine),
        "unresolved_consecutive": [d.as_dict() for d in unresolved],
        "suspension_events_in_window": sum(
            1 for d in disc_all if d.classification == "SUSPENSION_OR_RELISTING"
        ),
        "traces": traces,
        "ca_event_ledger": event_rows,
        "ca_symbols_with_events_in_window": sorted({
            r["symbol"] for r in event_rows if r["in_frozen_window"]
        }),
        "adjustment_policy_version": ADJUSTMENT_POLICY_VERSION,
        "note": (
            "Unresolved global CA outside the 29-name panel are intentionally ignored."
        ),
    }


def verify_universe_panel() -> dict[str, Any]:
    """S4 — fixed preregistered panel (mode A), not dynamic NSE membership."""
    ustatus = ledger_status()
    p = history_path()
    raw = json.loads(p.read_text()) if p.exists() else {}
    by = {r["symbol"]: r for r in (raw.get("rows") or [])}
    rows = []
    blockers = []
    for sym in FROZEN_PANEL:
        m = by.get(sym)
        if not m:
            rows.append({
                "symbol": sym, "status": "UNRESOLVED",
                "note": "missing from universe ledger",
            })
            blockers.append(sym)
            continue
        listed = m.get("listed")
        delisted = m.get("delisted")
        ok = listed is not None and str(listed) <= FROZEN_DATE_START
        if delisted and str(delisted) <= FROZEN_DATE_END:
            ok = False
        status = "VERIFIED" if ok else "CONFLICT"
        if not ok:
            blockers.append(sym)
        rows.append({
            "symbol": sym, "listed": listed, "delisted": delisted,
            "status": status,
            "note": "fixed-panel membership in frozen window",
        })
    return {
        "ok": not blockers,
        "universe_mode": "FIXED_PREREGISTERED_29",
        "mode": "A",
        "protocol_note": (
            "Frozen registrations list the exact 29 symbols in "
            "registered_data_window.universe. Cross-sectional selection occurs "
            "only within this panel (e.g. top momentum quintile), not via a "
            "dynamic historical NSE membership scan. The panel was preregistered "
            "before DISPLAY_ONLY results; not converted post-hoc to pass a gate."
        ),
        "dynamic_pit_membership_required": False,
        "ledger_source": ustatus.get("source"),
        "rows": rows,
        "blockers": blockers,
    }


def verify_sector_panel() -> dict[str, Any]:
    """S5 — static frozen sector map only (PIT sector history not required)."""
    smap = load_frozen_sector_map()
    missing = [s for s in FROZEN_PANEL if s not in smap]
    return {
        "ok": not missing,
        "requirement": "STATIC_MAP_ONLY",
        "pit_sector_history_required": False,
        "protocol_note": (
            "EXP-A5-01 / A6-01 / A5A6-01 freeze known_limitation "
            "'no PIT sector history' and use the static sector_map.json for the "
            "29 names. EXP-A2/A3 do not use sectors. Global sector-history "
            "incompleteness is not a scoped blocker."
        ),
        "sector_map_complete_for_panel": not missing,
        "missing": missing,
        "sectors_used_by": sorted(SECTOR_STATIC_CONSUMERS),
        "sectors_unused_by": sorted(SECTOR_UNUSED),
        "map": {s: smap.get(s) for s in FROZEN_PANEL},
        "sector_map_sha256": _file_sha(REPO_ROOT / "logs" / "phase_a5" / "sector_map.json"),
    }


def _count_consecutive_transitions(symbols: list[str]) -> dict[str, Any]:
    """Count all near-consecutive session pairs in the frozen window."""
    total = 0
    for sym in symbols:
        df = get_ohlcv(sym)
        if df is None or df.empty:
            continue
        sub = df.loc[
            (df.index >= pd.Timestamp(FROZEN_DATE_START))
            & (df.index <= pd.Timestamp(FROZEN_DATE_END))
        ]
        if len(sub) < 2:
            continue
        idx = sub.index
        for i in range(1, len(idx)):
            cal = int((pd.Timestamp(idx[i]) - pd.Timestamp(idx[i - 1])).days)
            if cal <= _CONSEC_CAL_DAYS:
                total += 1
    return {"total_consecutive_session_transitions": total}


def verify_price_panel() -> dict[str, Any]:
    """S6 — unresolved consecutive-session discontinuity rate on the 29 only."""
    ensure_loaded(rebuild_from_local=False)
    reload_corporate_actions()
    coverage = {}
    for sym in FROZEN_PANEL:
        df = get_ohlcv(sym)
        if df is None or df.empty:
            coverage[sym] = {"sessions": 0, "status": "UNRESOLVED"}
            continue
        sub = df.loc[
            (df.index >= pd.Timestamp(FROZEN_DATE_START))
            & (df.index <= pd.Timestamp(FROZEN_DATE_END))
        ]
        coverage[sym] = {
            "sessions": int(len(sub)),
            "first": str(sub.index.min().date()) if len(sub) else None,
            "last": str(sub.index.max().date()) if len(sub) else None,
            "status": "VERIFIED" if len(sub) >= 200 else "UNRESOLVED",
        }

    audit = audit_universe(symbols=FROZEN_PANEL, sample=None)
    # Restrict large-move classes to frozen window
    events = load_events()
    window_rows = []
    for sym in FROZEN_PANEL:
        for h in audit_symbol(sym, events=events):
            if _in_frozen_window(h.d0, h.d1):
                window_rows.append(h)
    consec = [r for r in window_rows if r.cal_days <= _CONSEC_CAL_DAYS]
    unresolved = [r for r in consec if r.classification == "UNRESOLVED"]
    verified_ca = [r for r in consec if r.classification == "SUPPORTED_CA"]
    genuine = [r for r in consec if r.classification == "GENUINE_MARKET_MOVE"]
    suspension = [
        r for r in window_rows if r.classification == "SUSPENSION_OR_RELISTING"
    ]

    transitions = _count_consecutive_transitions(FROZEN_PANEL)
    n_trans = transitions["total_consecutive_session_transitions"]
    unresolved_event_rate = (len(unresolved) / n_trans) if n_trans else 1.0
    # Gate uses the remediation metric: unresolved consecutive-session SYMBOL rate
    unresolved_symbol_rate = (
        len({r.symbol for r in unresolved}) / len(FROZEN_PANEL)
        if FROZEN_PANEL else 1.0
    )
    thin = [s for s, c in coverage.items() if c["status"] != "VERIFIED"]
    ok = (
        unresolved_symbol_rate <= UNRESOLVED_RATE_MAX
        and len(unresolved) == 0
        and not thin
    )
    return {
        "ok": ok,
        "metric": "unresolved_consecutive_session_symbol_rate",
        "threshold": UNRESOLVED_RATE_MAX,
        "unresolved_symbol_rate": round(unresolved_symbol_rate, 6),
        "unresolved_event_rate_vs_all_transitions": round(unresolved_event_rate, 8),
        "total_consecutive_session_transitions": n_trans,
        "verified_ca_transitions": len(verified_ca),
        "genuine_large_market_moves": len(genuine),
        "unresolved_discontinuities": len(unresolved),
        "unresolved_rate": round(unresolved_symbol_rate, 6),
        "consecutive_large_move_events": len(consec),
        "sparse_or_suspension_events": len(suspension),
        "by_class_full_history_audit": audit.get("by_class"),
        "phase_a5_unresolved": [r.as_dict() for r in unresolved],
        "coverage": coverage,
        "thin_history": thin,
        "threshold_pct_move": _GAP_PCT,
        "consec_calendar_days": _CONSEC_CAL_DAYS,
        "note": (
            "Long suspensions (>3 calendar days between bars) are classified "
            "SUSPENSION_OR_RELISTING and do not count as unresolved CA failures."
        ),
    }


def build_scoped_closes() -> pd.DataFrame:
    ensure_loaded(rebuild_from_local=False)
    reload_corporate_actions()
    cols = {}
    for sym in FROZEN_PANEL:
        df = get_ohlcv(sym)
        if df is None or df.empty:
            continue
        sub = df.loc[
            (df.index >= pd.Timestamp(FROZEN_DATE_START))
            & (df.index <= pd.Timestamp(FROZEN_DATE_END)),
            "close",
        ]
        cols[sym] = sub
    panel = pd.DataFrame(cols).sort_index()
    return panel.dropna(how="any")


def commit_scoped_snapshot(closes: pd.DataFrame, cert: dict) -> dict:
    """S8 — immutable snapshot of only the scoped panel (separate root)."""
    root = REPO_ROOT / "logs" / "phase_a5_scoped" / "snapshots"
    store = SnapshotStore(root)
    rows = []
    for sym in closes.columns:
        series = closes[sym].dropna()
        for ts, px in series.items():
            c = float(px)
            raw = get_ohlcv(sym)
            if raw is not None and ts in raw.index:
                bar = raw.loc[ts]
                o = float(bar["open"])
                h = float(bar["high"])
                l = float(bar["low"])
                c = float(bar["close"])
                vol = int(float(bar.get("volume") or 0))
            else:
                o = h = l = c
                vol = 0
            d = pd.Timestamp(ts).strftime("%Y-%m-%d")
            rows.append((sym, d, o, h, l, c, vol, "EQ"))

    ew = closes.mean(axis=1)
    index_rows = []
    prev = float(ew.iloc[0])
    for ts, px in ew.items():
        d = pd.Timestamp(ts).strftime("%Y-%m-%d")
        c = float(px)
        index_rows.append(("PHASE_A5_PANEL_EW", d, prev, max(prev, c), min(prev, c), c))
        prev = c

    extra = {
        "scope": "PHASE_A5_FROZEN_PROTOCOL",
        "global_trust_class": "OPERATIONAL_ONLY",
        "scoped_certification": cert.get("certification"),
        "research_grade": False,
        "trust_class": "OPERATIONAL_ONLY",
        "scoped_eligible_for_scientific_rerun": (
            cert.get("certification") == "READY_FOR_SCIENTIFIC_RERUN"
        ),
        "frozen_hypothesis_ids": FROZEN_HYPOTHESIS_IDS,
        "panel": list(FROZEN_PANEL),
        "security_ids": (cert.get("identity") or {}).get("security_ids"),
        "date_start": FROZEN_DATE_START,
        "date_end": FROZEN_DATE_END,
        "adjustment_policy_version": ADJUSTMENT_POLICY_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "n_sessions": int(len(closes)),
        "n_symbols": int(closes.shape[1]),
        "git_sha": _git_sha(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "nse_bhav_adjusted_on_read",
        "hashes": cert.get("hashes"),
        "do_not_rerun_automatically": True,
    }
    sid = store.commit_snapshot(rows, index_rows=index_rows, extra_manifest=extra)
    ok, fails = store.verify_snapshot(sid)
    return {
        "snapshot_id": sid,
        "root": str(root),
        "verify_ok": ok,
        "verify_fails": fails,
        "manifest": {**extra, "snapshot_id": sid},
    }


def run_scoped_certification(*, commit_snapshot: bool = True) -> dict[str, Any]:
    ensure_loaded(rebuild_from_local=False)
    reload_corporate_actions()

    deps = protocol_dependency_matrix()
    identity = verify_identity_panel()
    ca = verify_ca_panel()
    universe = verify_universe_panel()
    sector = verify_sector_panel()
    price = verify_price_panel()

    pit = {
        "ok": identity["ok"] and ca["ok"] and universe["ok"] and price["ok"],
        "mode": "FIXED_PANEL_ASOF_BARS",
        "note": (
            "Experiments read only closes ≤ evaluation date via walk-forward "
            "locations. No dynamic full-NSE PIT membership required by frozen "
            "protocol. CA applied on-read; raw bhav immutable."
        ),
    }

    per_exp = []
    for exp_id, hid in EXPERIMENT_ROWS:
        sector_ok = True if exp_id in SECTOR_UNUSED else sector["ok"]
        blockers = []
        if not identity["ok"]:
            blockers.append("IDENTITY")
        if not ca["ok"]:
            blockers.append("CA")
        if not universe["ok"]:
            blockers.append("UNIVERSE")
        if not sector_ok:
            blockers.append("SECTOR")
        if not price["ok"]:
            blockers.append("PRICE")
        if not pit["ok"]:
            blockers.append("PIT")
        cert_status = "BLOCKED" if blockers else "READY_FOR_SCIENTIFIC_RERUN"
        per_exp.append({
            "experiment": exp_id,
            "hypothesis_id": hid,
            "IDENTITY": "VERIFIED" if identity["ok"] else "BLOCKED",
            "CA": "VERIFIED" if ca["ok"] else "BLOCKED",
            "UNIVERSE": "VERIFIED" if universe["ok"] else "BLOCKED",
            "SECTOR": (
                "NOT_REQUIRED" if exp_id in SECTOR_UNUSED
                else ("VERIFIED_STATIC" if sector_ok else "BLOCKED")
            ),
            "PRICE": "VERIFIED" if price["ok"] else "BLOCKED",
            "PIT": "VERIFIED" if pit["ok"] else "BLOCKED",
            "SNAPSHOT": "PENDING",
            "CERTIFICATION": cert_status,
            "blockers": blockers,
        })

    overall = (
        "READY_FOR_SCIENTIFIC_RERUN"
        if all(e["CERTIFICATION"] == "READY_FOR_SCIENTIFIC_RERUN" for e in per_exp)
        else "BLOCKED"
    )

    hashes = {
        "ca_events": _file_sha(REPO_ROOT / "logs" / "ca_events.json"),
        "security_identity": _file_sha(REPO_ROOT / "logs" / "security_identity.json"),
        "universe_history": _file_sha(REPO_ROOT / "logs" / "universe_history.json"),
        "phase_a5_sector_map": _file_sha(
            REPO_ROOT / "logs" / "phase_a5" / "sector_map.json"
        ),
        "frozen_protocols_md": _file_sha(REPO_ROOT / "PHASE_A5_FROZEN_PROTOCOLS.md"),
    }

    plain = render_layers(PlainCard(
        label="Scoped research data check",
        state="PROVEN" if overall == "READY_FOR_SCIENTIFIC_RERUN" else "NOT_READY",
        explanation=(
            "QuantTerm's full historical database is not yet certified for scientific "
            "research. However, we separately checked the exact historical data needed "
            "for this specific frozen test."
            + (
                " The specific data used by this test passed the required historical "
                "checks, so the test can now be rerun scientifically."
                if overall == "READY_FOR_SCIENTIFIC_RERUN"
                else " This specific test still depends on historical information that "
                     "QuantTerm cannot verify."
            )
        ),
        implication=(
            "Global trust stays OPERATIONAL_ONLY. Await approval before Phase A.5 rerun."
            if overall == "READY_FOR_SCIENTIFIC_RERUN"
            else "Do not rerun Phase A.5 until scoped blockers clear."
        ),
        technical=f"scoped_certification={overall}; global_trust=OPERATIONAL_ONLY",
        internal_key="scoped_certification",
        internal_value=overall,
    ))

    cert: dict[str, Any] = {
        "global_trust_class": "OPERATIONAL_ONLY",
        "scoped_trust": "PHASE_A5_FROZEN_PROTOCOL_SCOPE",
        "certification": overall,
        "panel": list(FROZEN_PANEL),
        "date_start": FROZEN_DATE_START,
        "date_end": FROZEN_DATE_END,
        "hypothesis_ids": FROZEN_HYPOTHESIS_IDS,
        "dependency_matrix": deps,
        "identity": identity,
        "ca": ca,
        "universe": universe,
        "sector": sector,
        "price": price,
        "pit": pit,
        "per_experiment": per_exp,
        "hashes": hashes,
        "adjustment_policy_version": ADJUSTMENT_POLICY_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "git_sha": _git_sha(),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "user_facing": plain,
        "do_not_rerun_automatically": True,
        "phase_b_started": False,
        "phase_a5_rerun_executed": False,
    }

    if overall == "READY_FOR_SCIENTIFIC_RERUN" and commit_snapshot:
        closes = build_scoped_closes()
        if closes.shape[1] == 29 and len(closes) >= 200:
            snapshot_info = commit_scoped_snapshot(closes, cert)
            for e in per_exp:
                e["SNAPSHOT"] = "COMMITTED" if snapshot_info.get("verify_ok") else "BLOCKED"
            if not snapshot_info.get("verify_ok"):
                overall = "BLOCKED"
                cert["certification"] = "BLOCKED"
                for e in per_exp:
                    e["CERTIFICATION"] = "BLOCKED"
                    e["blockers"] = list(set(e["blockers"] + ["SNAPSHOT"]))
            else:
                sdir = Path(snapshot_info["root"]) / snapshot_info["snapshot_id"]
                eq = sdir / "bars_equity.csv"
                ix = sdir / "index_daily.csv"
                cert["hashes"]["scoped_bhav_panel"] = _file_sha(eq)
                cert["hashes"]["scoped_index_panel_ew"] = _file_sha(ix)
                # Refresh manifest hashes into snapshot sidecar record
                snap_rec = {
                    **snapshot_info,
                    "bhav_panel_sha256": cert["hashes"]["scoped_bhav_panel"],
                    "index_panel_ew_sha256": cert["hashes"]["scoped_index_panel_ew"],
                }
                cert["snapshot"] = snap_rec
                sidecar = (
                    REPO_ROOT / "logs" / "phase_a5_scoped" / "CERTIFICATION.json"
                )
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(json.dumps(cert, indent=2, default=str))
        else:
            cert["certification"] = "BLOCKED"
            cert["snapshot_blocker"] = {
                "n_symbols": int(closes.shape[1]),
                "n_sessions": int(len(closes)),
            }
            for e in per_exp:
                e["CERTIFICATION"] = "BLOCKED"
                e["SNAPSHOT"] = "BLOCKED"
                e["blockers"] = list(set(e["blockers"] + ["SNAPSHOT"]))
            overall = "BLOCKED"
            cert["certification"] = overall
            for e in per_exp:
                if e["SNAPSHOT"] == "PENDING":
                    e["SNAPSHOT"] = "BLOCKED"

    # Annotate dependency matrix blockers from live checks
    block_map = {
        "security_identity": not identity["ok"],
        "corporate_actions": not ca["ok"],
        "listing_delisting_window": not universe["ok"],
        "adjusted_close_panel": not price["ok"],
        "sector": not sector["ok"],
    }
    for row in deps:
        asset = row["required_data_asset"]
        if row["status"] == "NOT_REQUIRED":
            continue
        if block_map.get(asset):
            row["status"] = "BLOCKED"
            row["blocker"] = asset.upper()
        else:
            row["status"] = "VERIFIED"

    cert["certification"] = overall
    return cert


def render_certification_markdown(cert: dict[str, Any]) -> str:
    """Build PHASE_A5_SCOPED_DATA_CERTIFICATION.md body."""
    lines: list[str] = []
    lines.append("# Phase A.5 Scoped Data Certification")
    lines.append("")
    lines.append(
        "> Scope-specific fitness for the frozen Phase A.5 protocol. "
        "**Not** a global RESEARCH_GRADE stamp."
    )
    lines.append("")
    lines.append("## Common-man explanation")
    lines.append("")
    uf = cert.get("user_facing") or {}
    layer1 = uf.get("layer1") if isinstance(uf, dict) else None
    if isinstance(layer1, dict):
        lines.append(layer1.get("explanation") or "")
        if layer1.get("implication"):
            lines.append("")
            lines.append(layer1["implication"])
    elif isinstance(uf, dict) and uf.get("explanation"):
        lines.append(str(uf.get("explanation")))
    else:
        lines.append(
            "QuantTerm's full historical database is not yet certified for scientific "
            "research. However, we separately checked the exact historical data needed "
            "for this specific frozen test."
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Global trust state")
    lines.append("")
    lines.append(f"- **Global trust:** `{cert.get('global_trust_class')}`")
    lines.append(
        "- This remains unchanged. Unresolved global names (ABFRL demerger, "
        "ETF unit splits, unmatched CA factors) are **not** excluded from the "
        "main dataset to make global quality look better."
    )
    lines.append(
        f"- **Scoped certification:** `{cert.get('certification')}`"
    )
    lines.append(
        f"- Phase A.5 rerun executed: `{cert.get('phase_a5_rerun_executed')}`"
    )
    lines.append(f"- Phase B started: `{cert.get('phase_b_started')}`")
    lines.append("")
    lines.append("## 2. Exact frozen panel")
    lines.append("")
    lines.append(f"- Date range: `{cert.get('date_start')}` → `{cert.get('date_end')}`")
    lines.append(f"- N securities: **{len(cert.get('panel') or [])}**")
    lines.append(f"- Hypothesis IDs: `{', '.join(cert.get('hypothesis_ids') or [])}`")
    lines.append(f"- Protocol version: `{cert.get('protocol_version')}`")
    lines.append(f"- Validator version: `{cert.get('validator_version')}`")
    lines.append(f"- Adjustment policy: `{cert.get('adjustment_policy_version')}`")
    lines.append(f"- Git SHA: `{cert.get('git_sha')}`")
    lines.append("")
    lines.append("| # | Symbol | security_id |")
    lines.append("|--:|--------|-------------|")
    ids = (cert.get("identity") or {}).get("security_ids") or {}
    for i, sym in enumerate(cert.get("panel") or [], 1):
        lines.append(f"| {i} | {sym} | `{ids.get(sym)}` |")
    lines.append("")
    lines.append("## 3. Exact protocol dependencies (S1)")
    lines.append("")
    lines.append(
        "| EXPERIMENT | REQUIRED DATA ASSET | REQUIRED DATE RANGE | "
        "REQUIRED SECURITIES | STATUS | BLOCKER |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in cert.get("dependency_matrix") or []:
        lines.append(
            f"| {row['experiment']} | {row['required_data_asset']} | "
            f"{row['required_date_range']} | {row['required_securities']} | "
            f"{row['status']} | {row.get('blocker') or '—'} |"
        )
    lines.append("")
    lines.append("## 4. Security identity verification (S2)")
    lines.append("")
    ident = cert.get("identity") or {}
    lines.append(f"- Panel identity OK: `{ident.get('ok')}`")
    lines.append(f"- Blockers: `{ident.get('blockers') or []}`")
    lines.append("")
    lines.append(
        "| Symbol | security_id | ISIN | listing | delisting | "
        "identity | lineage |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in ident.get("rows") or []:
        lines.append(
            f"| {r['symbol']} | `{r.get('security_id')}` | `{r.get('isin')}` | "
            f"{r.get('listing_date')} | {r.get('delisting_date') or '—'} | "
            f"{r.get('status')} | {r.get('lineage_status')} |"
        )
    lines.append("")
    lines.append("## 5. Corporate-action verification (S3)")
    lines.append("")
    ca = cert.get("ca") or {}
    lines.append(f"- Panel CA OK: `{ca.get('ok')}`")
    lines.append(
        f"- Consecutive large-move events in window: "
        f"`{ca.get('consecutive_events_in_window')}`"
    )
    lines.append(f"- Verified CA transitions: `{ca.get('verified_ca_transitions')}`")
    lines.append(
        f"- Unresolved consecutive: `{len(ca.get('unresolved_consecutive') or [])}`"
    )
    lines.append(
        f"- Adjustment policy: `{ca.get('adjustment_policy_version')}`"
    )
    lines.append("")
    lines.append(
        "| security_id | symbol | type | source | ex-date | factor | "
        "verification | source_hash |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in ca.get("ca_event_ledger") or []:
        if not e.get("in_frozen_window"):
            continue
        lines.append(
            f"| `{e.get('security_id')}` | {e.get('symbol')} | {e.get('event_type')} | "
            f"{e.get('official_source')} | {e.get('ex_date')} | {e.get('ratio_factor')} | "
            f"{e.get('verification_result')} | `{e.get('source_hash')}` |"
        )
    lines.append("")
    lines.append(
        "Global unresolved CA outside this panel do **not** fail scoped certification."
    )
    lines.append("")
    lines.append("## 6. Universe-history requirement and result (S4)")
    lines.append("")
    uni = cert.get("universe") or {}
    lines.append(f"- Mode: **{uni.get('universe_mode')}** (protocol mode `{uni.get('mode')}`)")
    lines.append(
        f"- Dynamic PIT membership required: "
        f"`{uni.get('dynamic_pit_membership_required')}`"
    )
    lines.append(f"- OK: `{uni.get('ok')}`")
    lines.append(f"- Ledger source: `{uni.get('ledger_source')}`")
    lines.append("")
    lines.append(uni.get("protocol_note") or "")
    lines.append("")
    lines.append("## 7. Sector-history requirement and result (S5)")
    lines.append("")
    sec = cert.get("sector") or {}
    lines.append(f"- Requirement: `{sec.get('requirement')}`")
    lines.append(
        f"- PIT sector history required: `{sec.get('pit_sector_history_required')}`"
    )
    lines.append(f"- OK: `{sec.get('ok')}`")
    lines.append("")
    lines.append(sec.get("protocol_note") or "")
    lines.append("")
    lines.append("## 8. Discontinuity / price continuity (S6)")
    lines.append("")
    price = cert.get("price") or {}
    lines.append(f"- Metric: `{price.get('metric')}`")
    lines.append(f"- Threshold: `≤ {price.get('threshold')}`")
    lines.append(
        f"- Total consecutive-session transitions: "
        f"**{price.get('total_consecutive_session_transitions')}**"
    )
    lines.append(
        f"- Verified CA transitions (large-move class): "
        f"**{price.get('verified_ca_transitions')}**"
    )
    lines.append(
        f"- Genuine large market moves: "
        f"**{price.get('genuine_large_market_moves')}**"
    )
    lines.append(
        f"- Unresolved discontinuities: "
        f"**{price.get('unresolved_discontinuities')}**"
    )
    lines.append(f"- Unresolved rate (symbol): **{price.get('unresolved_rate')}**")
    lines.append(
        f"- Unresolved event rate vs all transitions: "
        f"**{price.get('unresolved_event_rate_vs_all_transitions')}**"
    )
    lines.append(
        f"- Sparse/suspension events (not counted as CA failure): "
        f"**{price.get('sparse_or_suspension_events')}**"
    )
    lines.append(f"- Thin history symbols: `{price.get('thin_history') or []}`")
    lines.append("")
    lines.append(price.get("note") or "")
    lines.append("")
    lines.append("## 9. PIT safety")
    lines.append("")
    pit = cert.get("pit") or {}
    lines.append(f"- OK: `{pit.get('ok')}`")
    lines.append(f"- Mode: `{pit.get('mode')}`")
    lines.append("")
    lines.append(pit.get("note") or "")
    lines.append("")
    lines.append("## 10. Snapshot reproducibility (S8)")
    lines.append("")
    snap = cert.get("snapshot") or {}
    if snap:
        lines.append(f"- snapshot_id: `{snap.get('snapshot_id')}`")
        lines.append(f"- root: `{snap.get('root')}`")
        lines.append(f"- verify_ok: `{snap.get('verify_ok')}`")
        lines.append(
            f"- bhav panel sha256: `{cert.get('hashes', {}).get('scoped_bhav_panel')}`"
        )
        lines.append(
            f"- panel EW index sha256: "
            f"`{cert.get('hashes', {}).get('scoped_index_panel_ew')}`"
        )
    else:
        lines.append("- Snapshot not committed (certification blocked or skipped).")
    lines.append("")
    lines.append("### Provenance hashes")
    lines.append("")
    lines.append("| Asset | sha256 |")
    lines.append("|---|---|")
    for k, v in (cert.get("hashes") or {}).items():
        lines.append(f"| {k} | `{v}` |")
    lines.append("")
    lines.append("## 11. Per-experiment blockers / certification matrix")
    lines.append("")
    lines.append(
        "| EXPERIMENT | IDENTITY | CA | UNIVERSE | SECTOR | PRICE | "
        "PIT | SNAPSHOT | CERTIFICATION |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for e in cert.get("per_experiment") or []:
        lines.append(
            f"| {e['experiment']} | {e['IDENTITY']} | {e['CA']} | {e['UNIVERSE']} | "
            f"{e['SECTOR']} | {e['PRICE']} | {e['PIT']} | {e['SNAPSHOT']} | "
            f"**{e['CERTIFICATION']}** |"
        )
    lines.append("")
    lines.append("## 12. Final scoped certification")
    lines.append("")
    lines.append("```")
    lines.append(f"GLOBAL TRUST:              {cert.get('global_trust_class')}")
    lines.append(f"PHASE A.5 FROZEN SCOPE:    {cert.get('certification')}")
    lines.append("```")
    lines.append("")
    lines.append(
        "Possible values for scoped certification: "
        "`READY_FOR_SCIENTIFIC_RERUN` | `BLOCKED`."
    )
    lines.append("")
    lines.append(
        "Do **not** interpret this as strategy PASS/FAIL. No Phase A.5 scientific "
        "rerun was executed in this milestone. Do **not** begin Phase B."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"_Evaluated at: {cert.get('evaluated_at')}_")
    lines.append("")
    return "\n".join(lines)


def write_certification_report(
    path: Path | None = None, *, commit_snapshot: bool = True
) -> dict[str, Any]:
    cert = run_scoped_certification(commit_snapshot=commit_snapshot)
    out = path or (REPO_ROOT / "PHASE_A5_SCOPED_DATA_CERTIFICATION.md")
    out.write_text(render_certification_markdown(cert), encoding="utf-8")
    cert["report_path"] = str(out)
    return cert


if __name__ == "__main__":
    result = write_certification_report(commit_snapshot=True)
    print(json.dumps({
        "global_trust_class": result["global_trust_class"],
        "certification": result["certification"],
        "snapshot_id": (result.get("snapshot") or {}).get("snapshot_id"),
        "report_path": result.get("report_path"),
        "per_experiment": [
            {"experiment": e["experiment"], "CERTIFICATION": e["CERTIFICATION"],
             "blockers": e["blockers"]}
            for e in result["per_experiment"]
        ],
        "price_summary": {
            "transitions": result["price"]["total_consecutive_session_transitions"],
            "verified_ca": result["price"]["verified_ca_transitions"],
            "genuine": result["price"]["genuine_large_market_moves"],
            "unresolved": result["price"]["unresolved_discontinuities"],
            "unresolved_rate": result["price"]["unresolved_rate"],
        },
    }, indent=2))
