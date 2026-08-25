"""Autonomy self-feed: taken vs skipped paper names, SEPA exam, cheap candidate tests.

This is the after-action blotter a desk would keep running all day — not a vanity
whole-market backtest. Official bhavcopy bars only. Missing levels or missing bars
are recorded as such. A single shadow WIN is not a sample size and is not a buy.

Paper behaviour change is conservative: a missed SEPA name may be *preferred among
already-generated paper signals*. SEPA never opens a position by itself. Live stays locked.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA_VERSION = 1
SHADOW_HORIZON = 10
MAX_SHADOW = 16
DEFAULT_LATEST = Path("logs/product/paper_self_feed.json")
DEFAULT_REPORT_DIR = Path("logs/autonomy/paper_cycle_reports")

NOT_A_SAMPLE = (
    "Candidate tests use official bars on names this desk already ranked. "
    "n_forward_bars is the window length, not a proven sample. Live stays locked."
)


def latest_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_PAPER_SELF_FEED")
    if override:
        return Path(override)
    return DEFAULT_LATEST


def reports_dir(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    override = os.environ.get("QT_PAPER_SELF_FEED_DIR")
    if override:
        return Path(override)
    return DEFAULT_REPORT_DIR


def _upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _as_dict(cycle: Any) -> dict[str, Any]:
    if cycle is None:
        return {}
    if isinstance(cycle, Mapping):
        return dict(cycle)
    as_dict = getattr(cycle, "as_dict", None)
    if callable(as_dict):
        try:
            return dict(as_dict() or {})
        except Exception:
            return {}
    return {}


def _pair_symbol(item: Any, *, symbol_index: int = 1) -> str:
    if isinstance(item, Mapping):
        return _upper(item.get("symbol"))
    if isinstance(item, (list, tuple)):
        if len(item) > symbol_index:
            return _upper(item[symbol_index])
        if item:
            return _upper(item[0])
    return _upper(item)


def _pair_reason(item: Any) -> str:
    if isinstance(item, Mapping):
        return str(item.get("reason") or item.get("reason_code") or item.get("why") or "")
    if isinstance(item, (list, tuple)) and len(item) >= 3:
        return str(item[2] or "")
    return ""


def _taken_symbols(cycle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(cycle.get("positions_opened") or []):
        symbol = _pair_symbol(item)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        strategy = ""
        if isinstance(item, (list, tuple)) and item:
            strategy = str(item[0] or "")
        elif isinstance(item, Mapping):
            strategy = str(item.get("strategy_id") or "")
        rows.append({"symbol": symbol, "strategy_id": strategy, "status": "taken"})
    return rows


def _skipped_rows(cycle: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in list(cycle.get("blocked_target_positions") or []):
        symbol = _pair_symbol(item)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        rows.append({
            "symbol": symbol,
            "status": "skipped",
            "reason": _pair_reason(item) or "BLOCKED_TARGET",
        })
    for item in list(cycle.get("signals_rejected") or []):
        symbol = _pair_symbol(item, symbol_index=0)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        rows.append({
            "symbol": symbol,
            "status": "skipped",
            "reason": _pair_reason(item) or "SIGNAL_REJECTED",
        })
    return rows


def load_sepa_best(scan: Mapping[str, Any] | None = None) -> tuple[list[dict[str, Any]], str]:
    """Persisted SEPA overlay only. Does not re-score OHLCV."""
    try:
        from product.scan_store import load_scan
        from product.sepa_setup import load_persisted_best_setups

        payload = dict(scan) if scan else (load_scan() or {})
        scanned_at = str(payload.get("scanned_at") or "")
        persisted = load_persisted_best_setups(scanned_at) if scanned_at else None
        if persisted is None:
            return [], "SEPA ranking file is missing or from a different scan."
        cards, note = persisted
        return list(cards or []), str(note or "SEPA overlay from the last saved scan.")
    except Exception as exc:
        return [], f"SEPA exam unavailable: {exc}"


def _col(frame: Any, *names: str) -> Any:
    cols = {str(c).lower(): c for c in getattr(frame, "columns", [])}
    for name in names:
        key = name.lower()
        if key in cols:
            return frame[cols[key]]
    raise KeyError(names[0] if names else "column")


def _forward_arrays(frame: Any, as_of: str, horizon: int) -> tuple[Any, Any, Any] | None:
    if frame is None or getattr(frame, "empty", True):
        return None
    stamp = str(as_of or "")[:10]
    index = getattr(frame, "index", None)
    if index is None:
        return None
    try:
        labels = [str(v)[:10] for v in index]
    except Exception:
        return None
    start = 0
    if stamp:
        for i, label in enumerate(labels):
            if label > stamp:
                start = i
                break
        else:
            return None
    window = frame.iloc[start:start + max(1, int(horizon))]
    if getattr(window, "empty", True):
        return None
    try:
        import numpy as np

        high = np.asarray(_col(window, "high", "High"), dtype=float)
        low = np.asarray(_col(window, "low", "Low"), dtype=float)
        close = np.asarray(_col(window, "close", "Close"), dtype=float)
    except Exception:
        return None
    if high.size == 0:
        return None
    return high, low, close


def shadow_setup(
    symbol: str,
    *,
    entry: Any,
    stop: Any,
    target: Any,
    as_of: str,
    load_frame: Callable[[str], Any] | None = None,
    horizon: int = SHADOW_HORIZON,
) -> dict[str, Any]:
    """First-touch test on official bars after as_of. Honest empties, no invented n."""
    row = {
        "symbol": _upper(symbol),
        "outcome": "NO_LEVELS",
        "r_multiple": None,
        "n_forward_bars": 0,
        "horizon": int(horizon),
        "note": NOT_A_SAMPLE,
    }
    try:
        entry_f = float(entry or 0)
        stop_f = float(stop or 0)
        target_f = float(target or 0)
    except (TypeError, ValueError):
        return row
    if entry_f <= 0 or stop_f <= 0 or target_f <= 0 or entry_f <= stop_f:
        return row
    loader = load_frame
    if loader is None:
        try:
            from data.bhavcopy_runtime import get_ohlcv as loader
        except Exception:
            row["outcome"] = "NO_BARS"
            return row
    try:
        frame = loader(_upper(symbol))
    except Exception:
        row["outcome"] = "NO_BARS"
        return row
    arrays = _forward_arrays(frame, as_of, horizon)
    if arrays is None:
        row["outcome"] = "NO_BARS"
        return row
    high, low, close = arrays
    row["n_forward_bars"] = int(len(close))
    try:
        from scan.signal_backtest import _simulate

        outcome, r_mult = _simulate(entry_f, stop_f, target_f, high, low, close)
    except Exception as exc:
        row["outcome"] = "TEST_FAILED"
        row["note"] = f"{NOT_A_SAMPLE} ({type(exc).__name__})"
        return row
    row["outcome"] = str(outcome or "FLAT")
    try:
        row["r_multiple"] = round(float(r_mult), 4)
    except (TypeError, ValueError):
        row["r_multiple"] = None
    return row


def _scan_levels(scan: Mapping[str, Any] | None, symbol: str) -> dict[str, Any]:
    key = _upper(symbol)
    for row in list((scan or {}).get("records") or []):
        if _upper(row.get("symbol")) == key:
            return dict(row)
    return {}


def _paper_status(symbol: str, taken: set[str], skipped: dict[str, str], cooldown: set[str]) -> str:
    key = _upper(symbol)
    if key in taken:
        return "taken"
    if key in cooldown:
        return "cooldown"
    if key in skipped:
        return "skipped"
    return "not_signaled"


def build_report(
    cycle: Any,
    *,
    scan: Mapping[str, Any] | None = None,
    sepa_cards: list[Mapping[str, Any]] | None = None,
    sepa_note: str = "",
    paper_memory: Mapping[str, Any] | None = None,
    as_of: str = "",
    slot: str = "intraday",
    load_frame: Callable[[str], Any] | None = None,
    horizon: int = SHADOW_HORIZON,
) -> dict[str, Any]:
    payload = _as_dict(cycle)
    as_of_day = str(as_of or payload.get("as_of_date") or "")[:10]
    taken = _taken_symbols(payload)
    skipped = _skipped_rows(payload)
    taken_set = {row["symbol"] for row in taken}
    skipped_map = {row["symbol"]: row["reason"] for row in skipped}
    cooldown = {
        _upper(row.get("symbol"))
        for row in list((paper_memory or {}).get("cooldown") or [])
        if _upper(row.get("symbol"))
    }
    cards = [dict(card) for card in (sepa_cards or [])]
    sepa_exam: list[dict[str, Any]] = []
    for card in cards[:8]:
        symbol = _upper(card.get("symbol"))
        if not symbol:
            continue
        status = _paper_status(symbol, taken_set, skipped_map, cooldown)
        sepa_exam.append({
            "symbol": symbol,
            "sepa_score": card.get("sepa_score"),
            "sepa_verdict": card.get("sepa_verdict"),
            "paper_status": status,
            "skip_reason": skipped_map.get(symbol, ""),
            "not_a_buy": True,
        })

    candidates: list[tuple[str, dict[str, Any], str]] = []
    for card in cards:
        symbol = _upper(card.get("symbol"))
        if symbol:
            candidates.append((symbol, dict(card), "sepa_best"))
    scan_by_symbol = {
        _upper(row.get("symbol")): dict(row)
        for row in list((scan or {}).get("records") or [])
        if _upper(row.get("symbol"))
    }
    for row in taken + skipped:
        symbol = row["symbol"]
        if any(symbol == existing[0] for existing in candidates):
            continue
        candidates.append((symbol, scan_by_symbol.get(symbol, {}), row["status"]))

    shadows: list[dict[str, Any]] = []
    for symbol, src, role in candidates[:MAX_SHADOW]:
        levels = src if src.get("entry") else _scan_levels(scan, symbol)
        test = shadow_setup(
            symbol,
            entry=levels.get("entry") or src.get("entry"),
            stop=levels.get("stop") or src.get("stop"),
            target=levels.get("target") or src.get("target"),
            as_of=as_of_day,
            load_frame=load_frame,
            horizon=horizon,
        )
        test["role"] = role
        test["paper_status"] = _paper_status(symbol, taken_set, skipped_map, cooldown)
        shadows.append(test)

    missed_wins = [
        row["symbol"]
        for row in shadows
        if row.get("role") == "sepa_best"
        and row.get("paper_status") in {"skipped", "not_signaled", "cooldown"}
        and str(row.get("outcome") or "") == "WIN"
        and int(row.get("n_forward_bars") or 0) > 0
        and row["symbol"] not in cooldown
    ]
    summary = (
        f"{len(taken)} paper taken, {len(skipped)} skipped, {len(sepa_exam)} SEPA best examined, "
        f"{len(shadows)} candidate test(s). {NOT_A_SAMPLE}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of_day,
        "slot": str(slot or "intraday"),
        "eligibility": str(payload.get("eligibility") or ""),
        "entry_block_reason": str(payload.get("entry_block_reason") or ""),
        "taken": taken,
        "skipped": skipped,
        "sepa_best": sepa_exam,
        "sepa_note": sepa_note or "SEPA overlay is research, not a buy.",
        "candidate_tests": shadows,
        "shadow_prefer": missed_wins,
        "summary": summary,
        "live_locked": True,
        "disclaimer": NOT_A_SAMPLE,
    }


def save_report(report: Mapping[str, Any], *, latest: str | Path | None = None,
                directory: str | Path | None = None) -> Path:
    payload = dict(report)
    target = latest_path(latest)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    day = str(payload.get("as_of") or "unknown")[:10] or "unknown"
    log_dir = reports_dir(directory)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"{day}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
    return target


def load_latest(path: str | Path | None = None) -> dict[str, Any]:
    target = latest_path(path)
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
            return {}
        return payload
    except Exception:
        return {}


def attach_to_paper_memory(memory: Mapping[str, Any] | None, report: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fold the blotter into paper memory. Does not unlock live. Does not rewrite BUY lists."""
    payload = dict(memory or {})
    feed = dict(report or {})
    cooldown = {
        _upper(row.get("symbol"))
        for row in list(payload.get("cooldown") or [])
        if _upper(row.get("symbol"))
    }
    shadow_prefer = [
        _upper(symbol)
        for symbol in list(feed.get("shadow_prefer") or [])
        if _upper(symbol) and _upper(symbol) not in cooldown
    ]
    payload["shadow_prefer"] = shadow_prefer
    payload["self_feed"] = {
        "as_of": feed.get("as_of") or "",
        "slot": feed.get("slot") or "",
        "summary": feed.get("summary") or "",
        "taken": list(feed.get("taken") or []),
        "skipped": list(feed.get("skipped") or []),
        "sepa_best": list(feed.get("sepa_best") or []),
        "candidate_tests": list(feed.get("candidate_tests") or []),
        "disclaimer": NOT_A_SAMPLE,
        "live_locked": True,
    }
    if shadow_prefer:
        extra = (
            f" Missed SEPA names with a WIN shadow preferred only if they already have a paper signal: "
            f"{', '.join(shadow_prefer)}."
        )
        payload["summary"] = str(payload.get("summary") or "") + extra
    return payload


def fold_latest_into_memory(memory: Mapping[str, Any] | None, *, path: str | Path | None = None) -> dict[str, Any]:
    from product.paper_learning import save_paper_memory

    merged = attach_to_paper_memory(memory, load_latest(path))
    save_paper_memory(merged)
    return merged


def ingest_paper_cycle(
    cycle: Any,
    *,
    as_of: str = "",
    slot: str = "intraday",
    scan: Mapping[str, Any] | None = None,
    sepa_cards: list[Mapping[str, Any]] | None = None,
    sepa_note: str = "",
    load_frame: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Write the blotter and update paper memory. Fail-open at the caller."""
    note = sepa_note
    cards = [dict(card) for card in sepa_cards] if sepa_cards is not None else None
    if cards is None:
        loaded, note = load_sepa_best(scan)
        cards = loaded
    if scan is None:
        try:
            from product.scan_store import load_scan
            scan = load_scan() or {}
        except Exception:
            scan = {}
    try:
        from product.paper_learning import load_paper_memory, save_paper_memory
        memory = load_paper_memory()
    except Exception:
        memory = {}
        save_paper_memory = None  # type: ignore[assignment]
    report = build_report(
        cycle,
        scan=scan,
        sepa_cards=cards,
        sepa_note=note,
        paper_memory=memory,
        as_of=as_of,
        slot=slot,
        load_frame=load_frame,
    )
    save_report(report)
    if save_paper_memory is not None:
        save_paper_memory(attach_to_paper_memory(memory, report))
    return report
