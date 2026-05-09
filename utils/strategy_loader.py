"""
Strategy loader + lightweight backtest runner.

Purpose
-------
Bridge the AlgoLab UI (where strategies are saved into
`algolab_strategies.db`) and the CLI (`main.py backtest --strategy NAME ...`),
without duplicating the heavy event-driven `Backtester` class (which is
designed around ML+LLM signals, not user-callable functions).

Two public helpers
------------------
- `load_strategy_from_db(name)`:
    Reads the strategy code by name and returns a callable
    `generate_signals(df: pd.DataFrame) -> pd.Series` (-1 / 0 / +1).

- `simulate_strategy(generate_signals_fn, df, capital)`:
    Runs the same vectorised simulation that the AlgoLab UI uses,
    but takes a callable instead of a code string. Returns a metrics
    dict identical to the UI's backtest result (sharpe, max_dd,
    win_rate, total_return, n_trades, equity_curve, trades).

Both helpers are pure Python — no Streamlit imports — so they are
safe to call from the CLI.
"""
from __future__ import annotations

import builtins
import sqlite3
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Repo root — `utils/` lives at /<repo>/utils/, DB at /<repo>/algolab_strategies.db
_REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = _REPO_ROOT / "algolab_strategies.db"


# ── Strategy loader ──────────────────────────────────────────────────────────
def load_strategy_from_db(strategy_name: str,
                          db_path: Path | str = DB_PATH) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Load `generate_signals` from a strategy stored in the AlgoLab SQLite DB.

    The schema is auto-detected (table name containing "strategy", columns
    containing "name" and one of {code, script, python_code, source}) so this
    helper survives small schema changes.
    """
    db = Path(db_path)
    if not db.exists():
        raise FileNotFoundError(
            f"Strategy DB not found at {db}. "
            "Save a strategy in the AlgoLab UI first."
        )

    conn = sqlite3.connect(db)
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        strategy_table = next((t for t in tables if "strateg" in t.lower()), None)
        if not strategy_table:
            raise RuntimeError(f"No strategy table found. Existing tables: {tables}")

        cur.execute(f"PRAGMA table_info({strategy_table})")
        col_names = [c[1] for c in cur.fetchall()]
        name_col = next((c for c in col_names if "name" in c.lower()), None)
        code_col = next((c for c in col_names
                         if c.lower() in ("code", "script", "python_code", "source")), None)
        if not name_col or not code_col:
            raise RuntimeError(
                f"Cannot identify name/code columns in `{strategy_table}`. "
                f"Columns: {col_names}"
            )

        cur.execute(
            f"SELECT {code_col} FROM {strategy_table} WHERE {name_col} = ?",
            (strategy_name,),
        )
        row = cur.fetchone()
    finally:
        conn.close()

    if not row:
        # List available names for the error message
        conn2 = sqlite3.connect(db)
        try:
            available = [r[0] for r in conn2.execute(
                f"SELECT {name_col} FROM {strategy_table}").fetchall()]
        finally:
            conn2.close()
        raise ValueError(
            f"Strategy '{strategy_name}' not found in DB. "
            f"Available: {available or 'none — save one in AlgoLab UI'}"
        )

    code = row[0]
    namespace: dict = {"pd": pd, "np": np, "__builtins__": builtins.__dict__}
    exec(compile(code, f"<strategy:{strategy_name}>", "exec"), namespace)  # noqa: S102
    fn = namespace.get("generate_signals")
    if not callable(fn):
        raise RuntimeError(
            f"Strategy '{strategy_name}' does not define `generate_signals(df)`."
        )
    return fn


# ── Lightweight simulation (mirrors AlgoLab UI's _run_backtest) ──────────────
def simulate_strategy(
    generate_signals: Callable[[pd.DataFrame], pd.Series],
    df: pd.DataFrame,
    capital: float = 1_000_000.0,
) -> dict:
    """
    Vectorised single-symbol backtest using a user-supplied
    `generate_signals(df) -> pd.Series` (values in {-1, 0, +1}).

    Returns a dict identical in shape to AlgoLab UI's backtest result:
      total_return (%), sharpe, max_dd (%), win_rate (%), n_trades,
      equity_curve [{date, equity}, ...], trades [{date, action, price, qty}, ...]
    """
    if df.empty or len(df) < 30:
        return {"error": "Not enough data (need ≥ 30 bars)"}

    try:
        signals = generate_signals(df.copy())
    except Exception as exc:  # noqa: BLE001 — surface user-strategy errors
        return {"error": f"strategy raised {type(exc).__name__}: {exc}"}

    if not isinstance(signals, pd.Series):
        return {"error": "generate_signals must return a pandas Series"}

    signals = signals.reindex(df.index).fillna(0).astype(int)
    cash, position, prev_sig = float(capital), 0, 0
    trades: list[dict] = []
    equity_curve: list[dict] = []

    for dt, row in df.iterrows():
        price = float(row["close"])
        sig   = int(signals.loc[dt])
        if sig == 1 and prev_sig != 1 and cash >= price:
            qty = int(cash * 0.95 / price)
            if qty:
                position += qty
                cash -= qty * price
                trades.append({"date": str(getattr(dt, "date", lambda: dt)()),
                               "action": "BUY", "price": price, "qty": qty})
        elif sig == -1 and prev_sig != -1 and position > 0:
            cash += position * price
            trades.append({"date": str(getattr(dt, "date", lambda: dt)()),
                           "action": "SELL", "price": price, "qty": position})
            position = 0
        equity = cash + position * price
        equity_curve.append({"date": str(getattr(dt, "date", lambda: dt)()),
                             "equity": equity})
        prev_sig = sig

    if not equity_curve:
        return {"error": "no equity points generated"}

    eq_arr  = np.asarray([e["equity"] for e in equity_curve], dtype=float)
    rets    = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([0.0])
    sharpe  = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak    = np.maximum.accumulate(eq_arr)
    dd_idx  = (eq_arr - peak).argmin()
    max_dd  = float((eq_arr[dd_idx] - peak[dd_idx]) / peak[dd_idx] * 100) \
              if peak[dd_idx] else 0.0
    wins = sum(1 for j in range(1, len(trades), 2)
               if j < len(trades) and trades[j]["price"] > trades[j-1]["price"])
    n_trades = len(trades) // 2
    win_rate = (wins / n_trades * 100) if n_trades else 0.0

    return {
        "total_return": (eq_arr[-1] - capital) / capital * 100,
        "sharpe":       round(sharpe, 2),
        "max_dd":       round(max_dd, 2),
        "win_rate":     round(win_rate, 1),
        "n_trades":     n_trades,
        "equity_curve": equity_curve,
        "trades":       trades,
    }
