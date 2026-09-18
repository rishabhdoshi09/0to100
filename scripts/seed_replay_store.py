"""Seed a deterministic local bhavcopy store so the FULL chain can be exercised
without market egress.

This is a development/replay harness. It is never a source of product evidence:
every bar is synthetic and callers must classify anything derived from it as
HISTORICAL_REPLAY or TEST_FIXTURE -- never PAPER_FORWARD.

Safety invariant: synthetic bars may only be written to an explicitly opted-in,
isolated fixture runtime. The adopted production runtime and any non-empty
unmarked runtime are refused before ``data.bhavcopy_store`` is imported.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_OPT_IN_ENV = "QT_ALLOW_SYNTHETIC_REPLAY_FIXTURE"
FIXTURE_MARKER = ".quantterm_synthetic_replay_fixture"


def _fixture_runtime_root() -> Path:
    """Return a proven fixture-only runtime root or raise before any market write."""
    from core.runtime_paths import ENV_VAR, REPO_ROOT, read_runtime_pointer, runtime_root

    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        raise RuntimeError(
            f"{ENV_VAR} must be explicitly set for the synthetic replay harness"
        )
    if os.environ.get(FIXTURE_OPT_IN_ENV, "").strip() != "1":
        raise RuntimeError(
            f"{FIXTURE_OPT_IN_ENV}=1 is required to create synthetic replay data"
        )

    root = runtime_root().expanduser().resolve()
    repo = REPO_ROOT.resolve()
    if root == repo or repo in root.parents:
        raise RuntimeError("synthetic replay runtime must be outside the source checkout")

    adopted = read_runtime_pointer()
    if adopted is not None and root == adopted.expanduser().resolve():
        raise RuntimeError(
            "refusing the adopted QuantTerm production runtime; choose an isolated fixture root"
        )

    # A new fixture root must be empty. Re-use is allowed only after this script
    # has marked it fixture-only. This prevents an explicit QT_RUNTIME_ROOT typo
    # from overwriting an existing production/history store.
    root.mkdir(parents=True, exist_ok=True)
    marker = root / FIXTURE_MARKER
    if not marker.exists():
        existing = [p for p in root.iterdir() if p.name != FIXTURE_MARKER]
        if existing:
            raise RuntimeError(
                "synthetic replay runtime is not empty and has no fixture marker"
            )
        marker.write_text(
            "QUANTTERM SYNTHETIC REPLAY FIXTURE ONLY\n"
            "Never use this runtime for production or PAPER_FORWARD evidence.\n",
            encoding="utf-8",
        )
    return root


def _sessions(n: int, end: date) -> list[pd.Timestamp]:
    out: list[pd.Timestamp] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(pd.Timestamp(d))
        d -= timedelta(days=1)
    return sorted(out)


def _series(kind: str, n: int, rng: np.random.Generator, base: float) -> np.ndarray:
    """Deterministic price paths with recognisable technical shapes."""
    t = np.arange(n)
    noise = rng.normal(0, 0.006, n).cumsum()
    if kind == "breakout":
        # long flat base, then a decisive expansion in the last ~12 sessions
        path = np.concatenate([np.zeros(n - 12), np.linspace(0, 0.22, 12)])
    elif kind == "uptrend":
        path = np.linspace(0, 0.45, n)
    elif kind == "pullback":
        path = np.concatenate([np.linspace(0, 0.38, n - 15), np.linspace(0.38, 0.27, 15)])
    elif kind == "downtrend":
        path = np.linspace(0, -0.32, n)
    elif kind == "extended":
        # already far above its mean -- the chase-risk / RSI guards should bite
        path = np.concatenate([np.linspace(0, 0.20, n - 20), np.linspace(0.20, 0.85, 20)])
    else:  # range
        path = 0.05 * np.sin(t / 7.0)
    return base * (1.0 + path + noise)


def build_store(
    symbols: int = 60,
    sessions: int = 260,
    end: date | None = None,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    end = end or date.today()
    days = _sessions(sessions, end)
    kinds = ["breakout", "uptrend", "pullback", "downtrend", "extended", "range"]
    store: dict[str, pd.DataFrame] = {}
    for i in range(symbols):
        sym = f"RPLY{i:03d}"
        kind = kinds[i % len(kinds)]
        base = float(rng.uniform(80, 2400))
        close = _series(kind, len(days), rng, base)
        # build a coherent OHLC around the close path
        intraday = np.abs(rng.normal(0, 0.008, len(days))) + 0.002
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        openp = np.concatenate([[close[0]], close[:-1]]) * (
            1 + rng.normal(0, 0.003, len(days))
        )
        openp = np.clip(openp, low, high)
        vol = rng.integers(80_000, 2_500_000, len(days)).astype(float)
        if kind == "breakout":
            vol[-12:] *= rng.uniform(2.0, 3.5)
        store[sym] = pd.DataFrame(
            {
                "date": days,
                "open": openp,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
                "deliv_per": rng.uniform(28, 78, len(days)),
            }
        )
    return store, days[-1].date(), days


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", type=int, default=60)
    ap.add_argument("--sessions", type=int, default=260)
    ap.add_argument("--end", type=str, default="")
    ap.add_argument(
        "--csv-sessions",
        type=int,
        default=60,
        help="raw CSV sessions to write; the historical bootstrap needs >=38",
    )
    args = ap.parse_args(argv)

    try:
        fixture_root = _fixture_runtime_root()
    except RuntimeError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    end = date.fromisoformat(args.end) if args.end else date.today()
    store, last_day, days = build_store(args.symbols, args.sessions, end)

    # Import only after fixture isolation has been proved. bhavcopy_store binds
    # its durable paths at import time, so this ordering is part of the safety
    # contract.
    import data.bhavcopy_store as BS

    BS._store = store
    BS._store_last_day = last_day
    BS._store_sessions = args.sessions
    BS._PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(BS._PKL, "wb") as fh:
        pickle.dump(
            {"store": store, "last_day": last_day, "sessions": args.sessions},
            fh,
        )

    # The historical-replay bootstrap reads raw per-session CSVs off disk
    # (_dates_on_disk), not the consolidated pickle, so a pickle-only store
    # leaves it permanently WAITING_FOR_HISTORY. Write the CSVs too.
    if args.csv_sessions:
        wanted = list(days)[-int(args.csv_sessions):]
        frames = {sym: df.set_index("date") for sym, df in store.items()}
        for ts in wanted:
            rows = []
            for sym, df in frames.items():
                if ts not in df.index:
                    continue
                r = df.loc[ts]
                rows.append(
                    {
                        "SYMBOL": sym,
                        "SERIES": " EQ",
                        "OPEN_PRICE": round(float(r["open"]), 2),
                        "HIGH_PRICE": round(float(r["high"]), 2),
                        "LOW_PRICE": round(float(r["low"]), 2),
                        "CLOSE_PRICE": round(float(r["close"]), 2),
                        "TTL_TRD_QNTY": int(r["volume"]),
                        "DELIV_PER": round(float(r["deliv_per"]), 2),
                    }
                )
            out = BS._BHAV_DIR / f"{ts.date().strftime('%d%m%Y')}.csv"
            pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  csv sessions : {len(wanted)} written to {BS._BHAV_DIR}")

    print(f"replay store seeded: {len(store)} symbols x {args.sessions} sessions")
    print(f"  fixture root : {fixture_root}")
    print(f"  last session : {last_day}")
    print(f"  cache        : {BS._PKL}")
    print("  EVIDENCE CLASS: HISTORICAL_REPLAY / TEST_FIXTURE -- never PAPER_FORWARD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
