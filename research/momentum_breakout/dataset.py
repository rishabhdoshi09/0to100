"""
📦 Dataset plumbing for the EXP-006 historical evidence run.

Three responsibilities, all fail-closed:
  • a DATA PROVIDER abstraction so the runner is data-source agnostic (the real
    NSE-bhav provider for the operator run; an injectable synthetic provider for
    deterministic, network-free tests);
  • a DATA-QUALITY GATE that produces a machine-readable report and refuses to run
    on corrupted price records or proven temporal violations;
  • a DATASET SNAPSHOT MANIFEST so a result is reproducible from stable identities.

Honest by construction: unavailable values stay unavailable (never coerced to 0),
and the report states every limitation (survivorship, sector membership, valuation,
delivery) rather than papering over it.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from research.momentum_breakout.config import MomentumBreakoutConfig, primary_config


class DatasetUnavailable(Exception):
    """Raised when the required point-in-time dataset does not exist / cannot be
    loaded. The runner turns this into an INCONCLUSIVE(DATA_UNAVAILABLE) verdict —
    never a fabricated PASS/FAIL."""


def code_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5,
                             cwd=str(Path(__file__).resolve().parent.parent.parent))
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════════
# Provider protocol (duck-typed) — a provider must expose:
#   calendar()          -> list[str] ISO dates, chronological (the benchmark cal)
#   benchmark_close()   -> np.ndarray aligned to calendar
#   benchmark_id()      -> str
#   symbols()           -> list[str]
#   ohlcv(sym)          -> dict{open,high,low,close,volume[,deliv]} np arrays aligned
#                          to calendar (NaN where the symbol did not trade), or None
#   sector_ctx(sym, i)  -> dict | None   (point-in-time; membership_pit flag inside)
#   valuation(sym, i)   -> dict | None   (must carry available_ts or be None)
#   source_identities() -> dict
#   universe_policy()   -> dict   (must include survivorship_complete: bool)
#   adjustment_policy() -> dict
# ══════════════════════════════════════════════════════════════════════════════

class BhavDataProvider:
    """The real operator provider: NSE official bhavcopy (CA-adjusted on read) +
    NSE index store benchmark + point-in-time universe. Fails closed if the store
    is empty and cannot be built (e.g. no network in a sandbox)."""

    def __init__(self, max_symbols: int | None = None, benchmark: str = "^NSEI"):
        import pandas as pd
        from data.bhavcopy_store import is_ready, build_store, store_symbols, get_ohlcv
        from data.index_store import get_index_ohlcv
        if not is_ready():
            try:
                build_store()
            except Exception as exc:
                raise DatasetUnavailable(f"bhav store build failed: {exc}")
        if not is_ready():
            raise DatasetUnavailable("bhav store empty and could not be built "
                                     "(no point-in-time NSE data available)")
        ndf = get_index_ohlcv(benchmark)
        col = next((c for c in ("Close", "close")
                    if ndf is not None and c in ndf.columns), None)
        if col is None:
            raise DatasetUnavailable(f"benchmark {benchmark} unavailable")
        nclose = ndf[col]
        self._cal = pd.DatetimeIndex(nclose.index)
        self._dates = [d.date().isoformat() for d in self._cal]
        self._bench = nclose.to_numpy(dtype=float)
        self._benchmark = benchmark
        self._get_ohlcv = get_ohlcv
        syms = store_symbols()
        self._symbols = sorted(syms)[:max_symbols] if max_symbols else sorted(syms)
        self._pd = pd
        # survivorship awareness (as-of the last calendar day)
        try:
            from data.nse_universe import point_in_time_universe
            self._universe = point_in_time_universe(self._dates[-1])
        except Exception:
            self._universe = {"survivorship_complete": False,
                              "note": "point_in_time_universe unavailable"}

    def calendar(self): return list(self._dates)
    def benchmark_close(self): return self._bench
    def benchmark_id(self): return self._benchmark
    def symbols(self): return list(self._symbols)

    def ohlcv(self, sym):
        df = self._get_ohlcv(sym)
        if df is None or "close" not in df.columns:
            return None
        out = {}
        for k in ("open", "high", "low", "close", "volume"):
            out[k] = (df[k].reindex(self._cal).to_numpy(dtype=float)
                      if k in df.columns else np.full(len(self._cal), np.nan))
        if "deliv_per" in df.columns:
            out["deliv"] = df["deliv_per"].reindex(self._cal).to_numpy(dtype=float)
        return out

    def sector_ctx(self, sym, i):
        # sector membership is NOT historically dated in the repo → membership_pit False.
        # We do not synthesise a survivorship-safe historical breadth we cannot support.
        return None

    def valuation(self, sym, i):
        # no point-in-time fundamentals with publication dates exist in the repo
        return None

    def source_identities(self):
        return {"prices": "NSE_bhavcopy_official_EOD_CA_adjusted_on_read",
                "benchmark": self._benchmark, "index_source": "NSE_index_store",
                "sector_source": "scan/sector_heat (current membership, NOT dated)",
                "fundamental_source": "NONE (no PIT fundamentals)"}

    def universe_policy(self):
        return {
            "survivorship_complete": bool(self._universe.get("survivorship_complete")),
            "research_grade": bool(self._universe.get("research_grade")),
            "source": str(self._universe.get("source") or ""),
            "note": self._universe.get("note", ""),
        }

    def adjustment_policy(self):
        from data.corporate_actions import load_events

        events = load_events()
        if events:
            return {
                "corporate_actions": "ADJUSTED",
                "mode": "adjust_on_read",
                "event_symbols": len(events),
            }
        return {
            "corporate_actions": "RAW",
            "mode": "unadjusted",
            "event_symbols": 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Data-quality gate
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityReport:
    ok: bool
    fatal_reasons: list = field(default_factory=list)
    limitations: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def as_dict(self):
        return {"ok": self.ok, "fatal_reasons": self.fatal_reasons,
                "limitations": self.limitations, "metrics": self.metrics}


def data_quality_report(provider, cfg: MomentumBreakoutConfig | None = None
                        ) -> DataQualityReport:
    """Machine-readable data-quality assessment. Fails closed (ok=False) on proven
    corruption (non-positive prices, high<low, duplicate dates) that could
    materially invalidate the experiment. Records every limitation."""
    cfg = cfg or primary_config()
    try:
        cal = provider.calendar()
        bench = provider.benchmark_close()
        symbols = provider.symbols()
    except DatasetUnavailable as exc:
        return DataQualityReport(ok=False, fatal_reasons=[f"DATA_UNAVAILABLE: {exc}"])
    if not cal or not symbols:
        return DataQualityReport(ok=False, fatal_reasons=["DATA_UNAVAILABLE: empty dataset"])

    n_sessions = len(cal)
    dup_dates = n_sessions - len(set(cal))
    m = {"symbols": len(symbols), "sessions": n_sessions,
         "date_range": [cal[0], cal[-1]], "duplicate_calendar_dates": dup_dates,
         "benchmark_id": provider.benchmark_id(),
         "benchmark_missing": int(np.sum(~np.isfinite(np.asarray(bench, float)))),
         "missing_ohlcv_cells": 0, "non_positive_prices": 0, "hloc_inconsistencies": 0,
         "ca_gap_anomalies": 0, "ipo_boundaries": 0, "terminal_history": 0,
         "insufficient_history_excluded": 0, "symbols_scanned": 0, "total_rows": 0}
    fatal, lims = [], []
    if dup_dates > 0:
        fatal.append(f"CORRUPTED: {dup_dates} duplicate calendar dates")

    min_hist = max(252, cfg.base_max_len + cfg.trend_ma) + 5
    for sym in symbols:
        d = provider.ohlcv(sym)
        if d is None:
            continue
        c = np.asarray(d["close"], float); h = np.asarray(d["high"], float)
        lo = np.asarray(d["low"], float); o = np.asarray(d["open"], float)
        valid = np.isfinite(c)
        n_valid = int(np.sum(valid))
        m["total_rows"] += n_valid
        m["missing_ohlcv_cells"] += int(np.sum(~valid))
        # non-positive prices (proven corruption on the bars that DO exist)
        m["non_positive_prices"] += int(np.sum(valid & (c <= 0)))
        # HLOC consistency on valid bars
        both = valid & np.isfinite(h) & np.isfinite(lo) & np.isfinite(o)
        bad = both & ((h < lo) | (h < np.maximum(o, c)) | (lo > np.minimum(o, c)))
        m["hloc_inconsistencies"] += int(np.sum(bad))
        # CA gap anomaly: an unexplained >40% single-day jump (possible unadjusted split)
        cc = c[valid]
        if cc.size > 1:
            rets = np.abs(np.diff(cc) / cc[:-1])
            m["ca_gap_anomalies"] += int(np.sum(rets > 0.40))
        # IPO / terminal boundaries
        if n_valid:
            first = int(np.argmax(valid)); last = int(len(valid) - 1 - np.argmax(valid[::-1]))
            if first > 0:
                m["ipo_boundaries"] += 1
            if last < n_sessions - 1:
                m["terminal_history"] += 1
        if n_valid < min_hist:
            m["insufficient_history_excluded"] += 1
        else:
            m["symbols_scanned"] += 1

    if m["non_positive_prices"] > 0:
        fatal.append(f"CORRUPTED: {m['non_positive_prices']} non-positive prices")
    if m["hloc_inconsistencies"] > 0:
        fatal.append(f"CORRUPTED: {m['hloc_inconsistencies']} high/low/open/close inconsistencies")
    if m["symbols_scanned"] == 0:
        fatal.append("DATA_UNAVAILABLE: no symbol has sufficient history to scan")

    # limitations (not fatal, but must be recorded and may bound the verdict)
    up = provider.universe_policy()
    if not up.get("survivorship_complete"):
        lims.append("SURVIVORSHIP_INCOMPLETE: universe is today's survivors — "
                    "historical results are optimistically biased")
    elif up.get("research_grade") is False:
        lims.append(
            "SURVIVORSHIP_INFERRED: membership is bhav-bootstrap / non-official "
            f"(source={up.get('source') or 'unknown'}) — not research-grade; "
            "PASS remains blocked until an official listing/delisting archive is ingested"
        )
    adj = provider.adjustment_policy()
    if "RAW" in json.dumps(adj):
        lims.append("CA_ADJUSTMENT: corporate actions applied only if ca_events.json "
                    "present; otherwise raw prices (phantom split/bonus gaps possible)")
    lims.append("SECTOR_MEMBERSHIP_NOT_PIT: sector classification is not historically dated")
    lims.append("VALUATION_DATA_UNAVAILABLE: no point-in-time fundamentals with publication dates")
    if m["ca_gap_anomalies"] > 0:
        lims.append(f"CA_GAP_ANOMALIES: {m['ca_gap_anomalies']} unexplained >40% one-day "
                    "moves (possible unadjusted corporate actions)")

    return DataQualityReport(ok=not fatal, fatal_reasons=fatal, limitations=lims, metrics=m)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset snapshot manifest
# ══════════════════════════════════════════════════════════════════════════════

def snapshot_manifest(provider, cfg: MomentumBreakoutConfig | None = None,
                      quality: DataQualityReport | None = None) -> dict:
    """Freeze the exact dataset identity. `snapshot_id` is a deterministic hash of
    the source identities + date range + symbol/row counts + config hash, so the
    same data + config reproduces the same id."""
    cfg = cfg or primary_config()
    try:
        cal = provider.calendar(); symbols = provider.symbols()
        dr = [cal[0], cal[-1]] if cal else [None, None]
        n_sym = len(symbols)
    except DatasetUnavailable:
        cal, dr, n_sym = [], [None, None], 0
    rows = quality.metrics.get("total_rows") if quality else None
    identity = {
        "source_identities": _safe(provider, "source_identities"),
        "date_range": dr, "symbol_count": n_sym, "row_count": rows,
        "adjustment_policy": _safe(provider, "adjustment_policy"),
        "universe_policy": _safe(provider, "universe_policy"),
        "benchmark_identity": _safe(provider, "benchmark_id"),
        "config_hash": cfg.config_hash(),
    }
    sid = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str)
                         .encode()).hexdigest()[:16]
    return {"snapshot_id": sid, **identity,
            "sector_data_identity": "scan/sector_heat (current membership, not dated)",
            "fundamental_data_identity": "NONE (no PIT fundamentals)",
            "cost_model": {"roundtrip_pct": cfg.cost_pct_roundtrip,
                           "slippage_pct": cfg.slippage_pct,
                           "note": "modelled aggregate — NOT a broker contract-note replication"},
            "code_commit": code_commit(),
            "experiment_config_hash": cfg.config_hash()}


def _safe(provider, method):
    try:
        return getattr(provider, method)()
    except Exception as exc:
        return f"unavailable: {exc}"
