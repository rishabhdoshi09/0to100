"""Prospective shadow test of the pre-registered ranking challenger.

The walk-forward (scan/signal_backtest.confirmation_report) is historical
out-of-sample evidence. This is the forward half: every night, BEFORE any of
its outcomes exist, freeze how the pre-registered composite (lower ATR %,
nearer the 52-week high, less extended above the 50-DMA; 2026-11-19) and the
champion score would rank that session's scan candidates. Only once each
candidate's whole 10-session window has completed AND its bars are verified
is it settled. The test is the same block inference as the walk-forward:
per-session tie-corrected Spearman IC vs 10-session excess return, averaged
in non-overlapping 10-session NSE blocks, Student t, BH; >= 20 independent
blocks before any verdict.

Shadow only: no paper order, no ranking change, no policy write. Rows are
append-only and frozen; a session is recorded once.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

SCHEMA_VERSION = 1
HYPOTHESIS = "2026-11-19:low_atr+near_high+not_extended"
SHADOW_FILE = "challenger_shadow.jsonl"


def ledger_path() -> Path:
    from core.runtime_paths import logs_path

    return logs_path("product", SHADOW_FILE)


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def record_session(session: str, candidates: Iterable[Mapping[str, Any]],
                   ohlcv_fn: Callable[[str], Any], *, path: Path | None = None) -> dict[str, Any]:
    """Freeze the challenger's and champion's view of one session (idempotent)."""
    from scan.signal_backtest import _average_ranks, confirmation_features
    import pandas as pd

    path = path or ledger_path()
    session = str(session)[:10]
    if any(r.get("session") == session for r in _read(path)):
        return {"session": session, "recorded": 0, "reason": "already_recorded"}
    cut = pd.Timestamp(session)
    rows = []
    for c in candidates:
        sym = str(c.get("symbol") or "").upper()
        if not sym:
            continue
        frame = ohlcv_fn(sym)
        if frame is None or getattr(frame, "empty", True):
            continue
        hist = frame[frame.index <= cut]
        if hist.empty or hist.index[-1].normalize() != cut:
            continue  # no verified bar for the session itself: not frozen, never guessed
        feats = confirmation_features(hist)
        if feats is None:
            continue
        rows.append({"symbol": sym, "champion_score": c.get("score"), **feats})
    if len(rows) >= 3:
        parts = [_average_ranks([float(r[f]) for r in rows]) for f in ("low_atr", "near_high", "not_extended")]
        for i, r in enumerate(rows):
            r["composite"] = sum(p[i] for p in parts)
    frozen_at = datetime.now(timezone.utc).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({"schema_version": SCHEMA_VERSION, "hypothesis": HYPOTHESIS,
                                 "session": session, "frozen_at": frozen_at, **r}, default=str) + "\n")
    return {"session": session, "recorded": len(rows)}


def evaluate(ohlcv_fn: Callable[[str], Any], *, maturity: Callable[[str], str],
             horizon: int | None = None, path: Path | None = None,
             session_index: dict[str, int] | None = None) -> dict[str, Any]:
    """Settle matured sessions and test challenger vs champion (read-only)."""
    from product.learning_scorecard import MATURED
    from scan.signal_backtest import MIN_INDEPENDENT_SESSIONS, _session_ordinals, spearman_ic
    import pandas as pd

    if horizon is None:
        from core.evaluation_spec import OUTCOME_HORIZON_SESSIONS as horizon  # type: ignore[no-redef]
    rows = _read(path or ledger_path())
    frames: dict[str, Any] = {}

    def frame_of(sym: str):
        if sym not in frames:
            frames[sym] = ohlcv_fn(sym)
        return frames[sym]

    by_session: dict[str, list[dict]] = {}
    for r in rows:
        by_session.setdefault(str(r.get("session")), []).append(r)
    statuses = {s: maturity(s) for s in by_session}
    daily: dict[str, dict[str, float]] = {}
    for session, rs in sorted(by_session.items()):
        if statuses[session] != MATURED:
            continue
        cut = pd.Timestamp(session)
        fwd = []
        for r in rs:
            frame = frame_of(str(r["symbol"]))
            if frame is None or getattr(frame, "empty", True):
                continue
            after = frame[frame.index > cut]
            if len(after) < int(horizon):
                continue
            o, c = float(after["open"].iloc[0]), float(after["close"].iloc[int(horizon) - 1])
            if o > 0:
                fwd.append((r, c / o - 1.0))
        if len(fwd) < 5:
            continue
        mean = sum(x for _, x in fwd) / len(fwd)
        excess = [x - mean for _, x in fwd]
        ic = {}
        for key in ("composite", "champion_score"):
            xs = [r.get(key) for r, _ in fwd]
            if all(x is not None for x in xs):
                v = spearman_ic([float(x) for x in xs], excess)
                if v is not None:
                    ic[key] = v
        if ic:
            daily[session] = ic
    ordinals = session_index if session_index is not None else _session_ordinals(sorted(daily))
    out: dict[str, Any] = {
        "hypothesis": HYPOTHESIS, "status": "SHADOW_ONLY",
        "sessions_recorded": len(by_session),
        "sessions_matured": sum(1 for s in statuses.values() if s == MATURED),
        "sessions_with_ic": len(daily), "results": {},
    }
    for key in ("composite", "champion_score"):
        res: dict[str, Any] = {}
        if ordinals is None:
            res = {"verdict": "INSUFFICIENT EVIDENCE", "reason": "EXCHANGE_CALENDAR_UNAVAILABLE"}
        else:
            blocks: dict[int, list[float]] = {}
            for s, ics in daily.items():
                if key in ics and s in ordinals:
                    blocks.setdefault(ordinals[s] // max(1, int(horizon)), []).append(ics[key])
            means = [sum(v) / len(v) for _, v in sorted(blocks.items())]
            k = len(means)
            res["independent_sessions"] = k
            if k < MIN_INDEPENDENT_SESSIONS:
                res.update({"verdict": "INSUFFICIENT EVIDENCE",
                            "reason": f"{k} of {MIN_INDEPENDENT_SESSIONS} independent blocks"})
            else:
                m = sum(means) / k
                sd = math.sqrt(sum((x - m) ** 2 for x in means) / (k - 1))
                res["mean_ic"] = round(m, 4)
                if sd <= 1e-9:
                    res.update({"verdict": "INSUFFICIENT EVIDENCE", "reason": "DEGENERATE_ZERO_DISPERSION"})
                else:
                    t = m / (sd / math.sqrt(k))
                    from scipy.stats import t as student_t

                    p = float(2.0 * student_t.sf(abs(t), df=k - 1))
                    res.update({"t": round(t, 3), "p_two_sided": round(p, 6),
                                "verdict": "POSITIVE_IC" if p < 0.05 and m > 0 else
                                           "NEGATIVE_IC" if p < 0.05 else "NOT_DISTINGUISHABLE_FROM_ZERO"})
        out["results"][key] = res
    out["note"] = ("Prospective, frozen before outcomes. Shadow only; a POSITIVE_IC still has to pass the "
                   "champion/challenger gate together with the walk-forward confirmation before PAPER.")
    return out


def nightly(session: str | None = None) -> dict[str, Any]:
    """Record tonight's scan and evaluate matured sessions (learning cycle hook)."""
    from data import bhavcopy_runtime, bhavcopy_store
    from product.learning_scorecard import maturity_checker
    from product.scan_store import default_scan_path, load_scan

    bhavcopy_runtime.ensure_loaded(rebuild_from_local=False)
    if not bhavcopy_store.is_ready():
        return {"error": "OFFICIAL_STORE_NOT_LOADED"}
    scan = load_scan(default_scan_path()) or {}
    session = str(session or scan.get("as_of_session") or "")[:10]
    recorded = {"recorded": 0, "reason": "no_scan_session"}
    if len(session) == 10 and scan.get("records"):
        recorded = record_session(session, scan.get("records") or [], bhavcopy_store.get_ohlcv)
    report = evaluate(bhavcopy_store.get_ohlcv, maturity=maturity_checker())
    return {"recorded": recorded, "evaluation": report}
