"""Fail-open FEATURE-002 observe hook. Never mutates production cards."""
from __future__ import annotations

import copy
import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

from research.feature002.constants import (
    EXCHANGE,
    FEATURE_SET_VERSION,
    FORWARD_START_DATE,
    FORWARD_START_TS_IST,
    LEDGER_DIR,
    PRIMARY_SOURCE,
    PRODUCTION_RANK_VERSION,
    SHADOW_RANK_VERSION,
    candidate_set_id,
    event_id,
    protocol_hash,
)
from research.feature002.ledger import insert_candidate_set, insert_observation
from research.feature002.ranks import apply_shadow_ranks

HOOK_LOG = LEDGER_DIR / "hook_log.jsonl"


def _write_hook_event(event: dict[str, Any]) -> None:
    """Best-effort receipt. Must never break the scan worker."""
    try:
        LEDGER_DIR.mkdir(parents=True, exist_ok=True)
        row = dict(event)
        row.setdefault("ts", _ist_now_iso())
        with HOOK_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:
        pass

_enabled = True
_lock = threading.Lock()


def set_enabled(flag: bool) -> None:
    global _enabled
    _enabled = bool(flag)


def is_enabled() -> bool:
    return _enabled


def _ist_now_iso() -> str:
    try:
        from core.market_clock import now_ist
        return now_ist().isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def _session_date(cards: Sequence[Mapping[str, Any]]) -> str:
    try:
        from core.market_clock import today_ist
        return today_ist().isoformat()
    except Exception:
        return datetime.now().date().isoformat()


def _label_to_key() -> dict[str, str]:
    from scan.unified_scanner import SIGNAL_META
    return {meta[0]: key for key, meta in SIGNAL_META.items()}


def _families_of(card: Mapping[str, Any], label_map: Mapping[str, str]) -> list[str]:
    raw = card.get("signal_keys") or card.get("signals") or []
    out: list[str] = []
    for item in raw:
        s = str(item)
        if s in label_map.values():
            out.append(s)
        elif s in label_map:
            out.append(label_map[s])
    return list(dict.fromkeys(out))


def _primary_family(families: Sequence[str]) -> str | None:
    from scan.unified_scanner import SIGNAL_META
    ranked = sorted(
        (f for f in families if f in SIGNAL_META),
        key=lambda f: -int(SIGNAL_META[f][2]),
    )
    return ranked[0] if ranked else (families[0] if families else None)


def _sector_map() -> tuple[dict[str, str], str]:
    try:
        from data.nse_universe import get_nse_universe_by_sector
        inv: dict[str, str] = {}
        for sec, syms in (get_nse_universe_by_sector() or {}).items():
            for s in syms:
                inv[str(s).upper()] = str(sec)
        return inv, "nse_universe_by_sector.v1"
    except Exception:
        return {}, ""


def _regime() -> str:
    try:
        from core.regime_engine import compute_regime
        return str(getattr(compute_regime(), "market_regime", "") or "")
    except Exception:
        return ""


def _hist(symbol: str):
    try:
        from scan.bulk_fetcher import get_cached
        df = get_cached(symbol)
        if df is not None and len(df):
            return df
    except Exception:
        pass
    try:
        from data.bhavcopy_store import get_ohlcv
        return get_ohlcv(symbol)
    except Exception:
        return None


def _as_of_from_hist(hist) -> str | None:
    try:
        import pandas as pd
        return str(pd.Timestamp(hist.index[-1]).date())
    except Exception:
        return None


def _build_rs_table(symbols: Sequence[str], as_of: str):
    try:
        from research.sepa.config import DEFAULT_CONFIG
        from research.sepa.universe_pit import FastInvestable
        from research.sepa003.fastrs import FastRS

        frames = {}
        for sym in symbols:
            df = _hist(sym)
            if df is not None and len(df):
                frames[str(sym).upper()] = df
        if not frames:
            return None
        fast = FastInvestable(frames)
        return FastRS(fast, DEFAULT_CONFIG).table(as_of, list(frames))
    except Exception:
        return None


def _features_for(symbol: str, hist, rs_table) -> dict[str, Any]:
    from research.feature001.rs_features import compute_rs_features
    from research.feature001.trend_features import compute_trend_features

    trend = compute_trend_features(hist)
    rs = compute_rs_features(symbol, rs_table)
    return {"trend": trend, "rs": rs}


def build_shadow_records(
    cards: Sequence[Mapping[str, Any]],
    *,
    session_date: str,
    recorded_at: str,
    scan_cycle_id: str,
    source: str,
    regime_label: str = "",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Pure function: cards are not mutated. Used by tests and the hook."""
    detached = [copy.deepcopy(dict(c)) for c in cards]
    label_map = _label_to_key()
    sectors, sector_ver = _sector_map()
    symbols = [str(c.get("symbol") or "").upper() for c in detached if c.get("symbol")]
    hists = {sym: _hist(sym) for sym in symbols}
    as_ofs = [_as_of_from_hist(h) for h in hists.values() if h is not None]
    # Feature bars are last official/hist date. Session date stays the IST
    # scan calendar so weekend/Monday scans do not collide with Friday event_ids.
    hist_as_of = session_date
    if as_ofs:
        hist_as_of = max(as_ofs)
    rs_table = _build_rs_table(symbols, hist_as_of)

    rows: list[dict[str, Any]] = []
    fam_counter: Counter[str] = Counter()
    for card in detached:
        sym = str(card.get("symbol") or "").upper()
        if not sym:
            continue
        families = _families_of(card, label_map)
        for f in families:
            fam_counter[f] += 1
        hist = hists.get(sym)
        feat = _features_for(sym, hist, rs_table) if hist is not None else {
            "trend": {"available": False}, "rs": {"available": False},
        }
        if isinstance(feat, dict):
            feat = dict(feat)
            feat["hist_as_of"] = hist_as_of
        trend = feat.get("trend") or {}
        rs = feat.get("rs") or {}
        verdict = str(card.get("verdict") or "")
        would = verdict in ("BUY", "STRONG BUY") and not bool(card.get("chase_risk"))
        rows.append({
            "symbol": sym,
            "exchange": EXCHANGE,
            "session_date": session_date,
            "families": families,
            "primary_family": _primary_family(families),
            "production_score": card.get("score"),
            "production_verdict": verdict,
            "production_signals": list(card.get("signals") or []),
            "production_decision": "TAKEN" if would else "WATCH",
            "would_trade": would,
            "ready_status": None,
            "entry": card.get("entry"),
            "stop": card.get("stop"),
            "target": card.get("target"),
            "chase_risk": bool(card.get("chase_risk")),
            "n_structure_passed": trend.get("n_structure_passed"),
            "structure_pass": trend.get("structure_pass"),
            "pct_above_sma200": trend.get("pct_above_sma200"),
            "ma_spread_50_200_pct": trend.get("ma_spread_50_200_pct"),
            "rs_percentile": rs.get("rs_percentile"),
            "rs_score": rs.get("rs_score"),
            "regime_label": regime_label,
            "sector": sectors.get(sym),
            "sector_map_version": sector_ver or None,
            "feature_snapshot": feat,
            "data_quality": "ok" if trend.get("available") or rs.get("available") else "incomplete",
        })

    apply_shadow_ranks(rows)
    set_id = candidate_set_id(scan_cycle_id)
    ph = protocol_hash()
    for row in rows:
        row["event_id"] = event_id(row["session_date"], row["symbol"])
        row["candidate_set_id"] = set_id
        row["scan_cycle_id"] = scan_cycle_id
        row["recorded_at"] = recorded_at
        row["source"] = source
        row["feature_set_version"] = FEATURE_SET_VERSION
        row["production_rank_version"] = PRODUCTION_RANK_VERSION
        row["shadow_rank_version"] = SHADOW_RANK_VERSION
        row["protocol_hash"] = ph

    cset = {
        "candidate_set_id": set_id,
        "scan_cycle_id": scan_cycle_id,
        "session_date": session_date,
        "recorded_at": recorded_at,
        "n_candidates": len(rows),
        "family_composition": dict(fam_counter),
        "source": source,
        "feature_set_version": FEATURE_SET_VERSION,
        "protocol_hash": ph,
    }
    return cset, rows


def persist_shadow(cset: dict[str, Any], rows: Sequence[dict[str, Any]], *, path=None) -> dict[str, Any]:
    insert_candidate_set(cset, path=path)
    n_new = 0
    n_exist = 0
    n_refused = 0
    for row in rows:
        out = insert_observation(row, path=path)
        if out.get("status") == "inserted":
            n_new += 1
        elif out.get("status") == "exists":
            n_exist += 1
        elif out.get("status") == "pre_freeze_refused":
            n_refused += 1
    result = {"inserted": n_new, "exists": n_exist, "pre_freeze_refused": n_refused,
              "candidate_set_id": cset["candidate_set_id"],
              "n_rows": len(rows), "session_date": cset.get("session_date"),
              "source": cset.get("source")}
    _write_hook_event({"kind": "persist_result", **result})
    return result


def observe_production_scan(
    serialized: Sequence[Mapping[str, Any]] | None,
    *,
    source: str = PRIMARY_SOURCE,
    background: bool = True,
    path=None,
) -> dict[str, Any] | None:
    """Entry point from auto_scan. Deepcopies immediately. Never writes back."""
    if not is_enabled() or not serialized:
        _write_hook_event({
            "kind": "hook_skipped",
            "reason": "disabled" if not is_enabled() else "empty_serialized",
            "source": source,
            "n_cards": 0 if not serialized else len(serialized),
        })
        return None
    snapshot = copy.deepcopy(list(serialized))
    _write_hook_event({
        "kind": "hook_received",
        "source": source,
        "n_cards": len(snapshot),
        "background": background,
    })

    def _run() -> dict[str, Any]:
        session = _session_date(snapshot)
        if source == PRIMARY_SOURCE and session < FORWARD_START_DATE:
            out = {"skipped": "pre_freeze_session", "session_date": session}
            _write_hook_event({"kind": "pre_freeze_session", **out})
            return out
        recorded = _ist_now_iso()
        if source == PRIMARY_SOURCE and recorded < FORWARD_START_TS_IST:
            # Clock before protocol activation: still persist as test, not primary.
            src = "implementation_test"
        else:
            src = source
        cycle = f"{session}:{uuid.uuid4().hex[:12]}"
        cset, rows = build_shadow_records(
            snapshot,
            session_date=session,
            recorded_at=recorded,
            scan_cycle_id=cycle,
            source=src,
            regime_label=_regime(),
        )
        return persist_shadow(cset, rows, path=path)

    if not background:
        try:
            return _run()
        except Exception:
            return {"skipped": "observe_failed"}
    threading.Thread(target=_safe_run, args=(_run,), name="feature002-shadow",
                     daemon=True).start()
    return {"queued": True}


def _safe_run(fn) -> None:
    try:
        fn()
    except Exception as exc:
        _write_hook_event({"kind": "observe_failed", "error": str(exc)})
        try:
            from logger import get_logger
            get_logger(__name__).debug("feature002_shadow_failed")
        except Exception:
            pass
