"""Point-in-time investable universe — never project end-of-sample liquidity back."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from research.sepa.frames import iso_date, pit_universe, slice_as_of


def membership_hash(symbols: Sequence[str]) -> str:
    blob = ",".join(sorted({str(s).upper() for s in symbols}))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _turnover(df: pd.DataFrame, window: int = 20) -> float:
    if df is None or len(df) == 0:
        return 0.0
    close = pd.to_numeric(df["close"], errors="coerce")
    vol = (
        pd.to_numeric(df["volume"], errors="coerce")
        if "volume" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    t = (close * vol).tail(window)
    val = float(t.mean()) if len(t) else 0.0
    return val if val == val else 0.0


def membership_as_of(as_of, *, frames: Mapping[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    """Symbols believed listed on as_of. Inferred membership is PIT_DEGRADED."""
    info = pit_universe(as_of)
    symbols = [str(s).upper() for s in (info.get("symbols") or [])]
    source = str(info.get("source") or "")
    if frames is not None and (not symbols or source.startswith("bhav_") or source in {"", "bhav_inferred"}):
        # Restrict inferred membership to names that actually have a bar ≤ as_of.
        live = []
        cutoff = iso_date(as_of)
        for sym, df in frames.items():
            sliced = slice_as_of(df, as_of)
            if sliced is None or len(sliced) == 0:
                continue
            live.append(str(sym).upper())
        if live:
            symbols = sorted(set(live))
            source = source or "bhav_inferred"
    return {
        "as_of": iso_date(as_of),
        "symbols": sorted(set(symbols)),
        "source": source or "bhav_inferred",
        "universe_complete": bool(info.get("universe_complete")),
        "research_grade": bool(info.get("research_grade")),
        "note": info.get("note") or "",
        "hash": membership_hash(symbols),
    }


@dataclass
class UniverseSnapshot:
    as_of: str
    candidates: list[str]
    investable: list[str]
    reasons: dict[str, str] = field(default_factory=dict)
    exclusions: dict[str, int] = field(default_factory=dict)
    source: str = ""
    membership_hash: str = ""
    investable_hash: str = ""
    min_price: float = 20.0
    min_turnover: float = 5_000_000.0
    min_sessions: int = 260
    top_n: int | None = None
    rs_denominator: int = 0

    def to_meta(self, symbol: str | None = None) -> dict[str, Any]:
        sym = str(symbol or "").upper()
        return {
            "universe_date": self.as_of,
            "candidate_count": len(self.candidates),
            "investable_count": len(self.investable),
            "rs_denominator": self.rs_denominator or len(self.investable),
            "universe_source": self.source,
            "membership_hash": self.membership_hash,
            "investable_hash": self.investable_hash,
            "selection_reason": self.reasons.get(sym, "not_in_snapshot"),
            "top_n": self.top_n,
        }


def screen_investable_as_of(
    frames: Mapping[str, pd.DataFrame],
    as_of,
    *,
    min_price: float = 20.0,
    min_turnover: float = 5_000_000.0,
    min_sessions: int = 260,
    quarantined: set[str] | None = None,
    membership: Sequence[str] | None = None,
    top_n: int | None = None,
) -> UniverseSnapshot:
    """Investability using only bars with date ≤ as_of.

    ``top_n`` is an optional liquidity cap for sensitivity studies. Canonical
    SEPA RS uses ``top_n=None`` (full as-of investable set).
    """
    as_of_s = iso_date(as_of)
    qset = {s.upper() for s in (quarantined or set())}
    if membership is None:
        mem = membership_as_of(as_of, frames=frames)
        candidates = list(mem["symbols"])
        source = mem["source"]
        mhash = mem["hash"]
    else:
        candidates = sorted({str(s).upper() for s in membership})
        source = "caller"
        mhash = membership_hash(candidates)

    reasons: dict[str, str] = {}
    exclusions: dict[str, int] = {}
    scored: list[tuple[float, str]] = []

    def _bump(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for raw in candidates:
        sym = str(raw).upper()
        df = frames.get(sym)
        if df is None:
            # frames may be keyed mixed-case
            df = next((frames[k] for k in frames if str(k).upper() == sym), None)
        sliced = slice_as_of(df, as_of) if df is not None else None
        if sliced is None or len(sliced) == 0:
            reasons[sym] = "no_bars"
            _bump("no_bars")
            continue
        if len(sliced) < min_sessions:
            reasons[sym] = "short_history"
            _bump("short_history")
            continue
        missing = [c for c in ("open", "high", "low", "close") if c not in sliced.columns]
        if missing:
            reasons[sym] = "incomplete_ohlcv"
            _bump("incomplete_ohlcv")
            continue
        try:
            px = float(sliced["close"].iloc[-1])
        except Exception:
            reasons[sym] = "bad_close"
            _bump("bad_close")
            continue
        if px < min_price:
            reasons[sym] = "min_price"
            _bump("min_price")
            continue
        to = _turnover(sliced)
        if to < min_turnover:
            reasons[sym] = "min_turnover"
            _bump("min_turnover")
            continue
        if sym in qset:
            reasons[sym] = "ca_quarantine"
            _bump("ca_quarantine")
            continue
        reasons[sym] = "investable"
        scored.append((to, sym))

    scored.sort(reverse=True)
    if top_n is not None and len(scored) > int(top_n):
        dropped = scored[int(top_n):]
        scored = scored[: int(top_n)]
        for _, sym in dropped:
            reasons[sym] = "not_top_n"
            _bump("not_top_n")
    investable = [sym for _, sym in scored]
    return UniverseSnapshot(
        as_of=as_of_s,
        candidates=list(candidates),
        investable=investable,
        reasons=reasons,
        exclusions=exclusions,
        source=source,
        membership_hash=mhash,
        investable_hash=membership_hash(investable),
        min_price=min_price,
        min_turnover=min_turnover,
        min_sessions=min_sessions,
        top_n=top_n,
        rs_denominator=len(investable),
    )


def load_store_frames(*, min_bars: int = 40) -> dict[str, pd.DataFrame]:
    """Every symbol with enough raw bars. No end-of-sample liquidity filter."""
    from data.bhavcopy_runtime import ensure_loaded
    from data.bhavcopy_store import get_ohlcv, store_symbols

    ensure_loaded(rebuild_from_local=False)
    out: dict[str, pd.DataFrame] = {}
    for sym in store_symbols() or []:
        df = get_ohlcv(sym)
        if df is None or len(df) < min_bars:
            continue
        out[str(sym).upper()] = df
    return out


class FastInvestable:
    """Precomputed close/turnover arrays so each as-of screen is O(symbols)."""

    def __init__(self, frames: Mapping[str, pd.DataFrame], *, turnover_window: int = 20):
        self.turnover_window = int(turnover_window)
        self._pos: dict[str, int] = {}
        self.symbols: list[str] = []
        self._dates: list[np.ndarray] = []
        self._close: list[np.ndarray] = []
        self._turn: list[np.ndarray] = []
        self._frames = {str(s).upper(): df.sort_index() for s, df in frames.items()}
        for sym, df in self._frames.items():
            if df is None or len(df) == 0 or "close" not in df.columns:
                continue
            idx = pd.DatetimeIndex(df.index)
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_localize(None)
            idx = idx.normalize()
            close = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
            vol = (
                pd.to_numeric(df["volume"], errors="coerce").to_numpy(dtype=float)
                if "volume" in df.columns
                else np.zeros(len(df))
            )
            turn = close * vol
            # rolling mean of last `window` sessions, NaN until full
            csum = np.cumsum(np.nan_to_num(turn, nan=0.0))
            win = int(turnover_window)
            roll = np.full(len(turn), np.nan)
            if len(turn) >= win:
                roll[win - 1:] = (csum[win - 1:] - np.concatenate([[0.0], csum[:-win]])) / win
            self._pos[sym] = len(self.symbols)
            self.symbols.append(sym)
            self._dates.append(idx.asi8)
            self._close.append(close)
            self._turn.append(roll)

    def hist_fwd(self, symbol: str, as_of, horizon: int):
        """Return (hist_df, fwd_df) using only bars ≤ as_of for hist."""
        from research.sepa.frames import iso_date
        import pandas as pd
        sym = str(symbol).upper()
        i = self._pos.get(sym)
        if i is None:
            return None, None
        as_ns = int(pd.Timestamp(iso_date(as_of)).normalize().value)
        j = self.loc_as_of(self._dates[i], as_ns)
        if j < 0:
            return None, None
        df = self._frames[sym]
        hist = df.iloc[: j + 1]
        fwd = df.iloc[j + 1: j + 1 + int(horizon)]
        return hist, fwd if len(fwd) else None

    def frame(self, symbol: str) -> pd.DataFrame | None:
        return self._frames.get(str(symbol).upper())

    def loc_as_of(self, dates: np.ndarray, as_of_ns: int) -> int:
        """Last index with date ≤ as_of. -1 if none."""
        i = int(np.searchsorted(dates, as_of_ns, side="right") - 1)
        return i

    def snapshot(
        self,
        as_of,
        *,
        min_price: float = 20.0,
        min_turnover: float = 5_000_000.0,
        min_sessions: int = 260,
        quarantined: set[str] | None = None,
        top_n: int | None = None,
        source: str = "bhav_inferred",
        mem_hash: str = "",
    ) -> UniverseSnapshot:
        as_of_s = iso_date(as_of)
        as_ns = int(pd.Timestamp(as_of_s).normalize().value)
        qset = {s.upper() for s in (quarantined or set())}
        reasons: dict[str, str] = {}
        exclusions: dict[str, int] = {}
        scored: list[tuple[float, str]] = []

        def bump(key: str) -> None:
            exclusions[key] = exclusions.get(key, 0) + 1

        candidates = list(self.symbols)
        for i, sym in enumerate(self.symbols):
            j = self.loc_as_of(self._dates[i], as_ns)
            if j < 0:
                reasons[sym] = "no_bars"
                bump("no_bars")
                continue
            n = j + 1
            if n < min_sessions:
                reasons[sym] = "short_history"
                bump("short_history")
                continue
            px = float(self._close[i][j])
            if not (px == px) or px < min_price:
                reasons[sym] = "min_price"
                bump("min_price")
                continue
            to = float(self._turn[i][j])
            if not (to == to) or to < min_turnover:
                reasons[sym] = "min_turnover"
                bump("min_turnover")
                continue
            if sym in qset:
                reasons[sym] = "ca_quarantine"
                bump("ca_quarantine")
                continue
            reasons[sym] = "investable"
            scored.append((to, sym))
        scored.sort(reverse=True)
        if top_n is not None and len(scored) > int(top_n):
            for _, sym in scored[int(top_n):]:
                reasons[sym] = "not_top_n"
                bump("not_top_n")
            scored = scored[: int(top_n)]
        investable = [s for _, s in scored]
        return UniverseSnapshot(
            as_of=as_of_s,
            candidates=candidates,
            investable=investable,
            reasons=reasons,
            exclusions=exclusions,
            source=source,
            membership_hash=mem_hash or membership_hash(candidates),
            investable_hash=membership_hash(investable),
            min_price=min_price,
            min_turnover=min_turnover,
            min_sessions=min_sessions,
            top_n=top_n,
            rs_denominator=len(investable),
        )

