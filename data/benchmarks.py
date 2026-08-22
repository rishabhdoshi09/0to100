"""Official NSE index series from local `ind_close_all` CSVs.

Never hits the network. Distinguishes price-return vs total-return by name.
A missing benchmark return is UNKNOWN, never 0.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

_DIR = Path(__file__).resolve().parent.parent / "logs" / "indices"

# Official names as printed in NSE ind_close_all_*.csv
PRICE_RETURN = {
    "Nifty 50": "NIFTY50",
    "Nifty 500": "NIFTY500",
    "Nifty Total Market": "NIFTY_TOTAL_MARKET",
    "Nifty 100": "NIFTY100",
    "Nifty Bank": "NIFTY_BANK",
    "Nifty IT": "NIFTY_IT",
    "Nifty Pharma": "NIFTY_PHARMA",
    "Nifty FMCG": "NIFTY_FMCG",
    "Nifty Auto": "NIFTY_AUTO",
    "Nifty Metal": "NIFTY_METAL",
    "Nifty Energy": "NIFTY_ENERGY",
    "Nifty Realty": "NIFTY_REALTY",
    "Nifty Smallcap 100": "NIFTY_SMALLCAP100",
    "India VIX": "INDIA_VIX",
}
TOTAL_RETURN_HINTS = (" TR ", "TR Index", "Total Return")

ALIASES = {
    "NIFTY50": "Nifty 50",
    "^NSEI": "Nifty 50",
    "NIFTY500": "Nifty 500",
    "^NSE500": "Nifty 500",
    "NIFTY_TOTAL_MARKET": "Nifty Total Market",
    "^CNXIT": "Nifty IT",
    "^NSEBANK": "Nifty Bank",
}


def _parse_stem(name: str):
    try:
        return datetime.strptime(Path(name).stem, "%d%m%Y").date()
    except ValueError:
        return None


def _return_kind(official_name: str) -> str:
    n = official_name
    if any(h in n for h in TOTAL_RETURN_HINTS):
        return "total_return"
    return "price_return"


def list_local_files(index_dir: Path | None = None) -> list[Path]:
    d = Path(index_dir) if index_dir is not None else _DIR
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.csv") if _parse_stem(p.name))


def file_coverage(index_dir: Path | None = None) -> dict[str, Any]:
    files = list_local_files(index_dir)
    dates = [_parse_stem(p.name) for p in files]
    dates = [d for d in dates if d]
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(str(p.stat().st_size).encode())
    return {
        "n_files": len(files),
        "first": str(min(dates)) if dates else None,
        "last": str(max(dates)) if dates else None,
        "source": "nsearchives.nseindia.com/content/indices/ind_close_all_DDMMYYYY.csv",
        "files_hash": h.hexdigest()[:16] if files else None,
        "network": False,
    }


def load_index(name: str, *, as_of: str | None = None, index_dir: Path | None = None) -> dict[str, Any]:
    """OHLC for one official index from local CSVs only."""
    import pandas as pd

    official = ALIASES.get(name, name)
    frames = []
    for p in list_local_files(index_dir):
        d = _parse_stem(p.name)
        if d is None:
            continue
        if as_of and str(d) > str(as_of)[:10]:
            continue
        try:
            df = pd.read_csv(p, dtype=str)
            df.columns = [c.strip() for c in df.columns]
            if "Index Name" not in df.columns:
                continue
            hit = df[df["Index Name"].str.strip() == official]
            if hit.empty:
                continue
            row = hit.iloc[0]
            frames.append({
                "date": str(d),
                "open": _num(row.get("Open Index Value")),
                "high": _num(row.get("High Index Value")),
                "low": _num(row.get("Low Index Value")),
                "close": _num(row.get("Closing Index Value")),
            })
        except Exception:
            continue
    frames.sort(key=lambda r: r["date"])
    return {
        "name": official,
        "alias": name,
        "return_kind": _return_kind(official),
        "source": "nse_ind_close_all_local",
        "rows": frames,
        "n": len(frames),
        "first": frames[0]["date"] if frames else None,
        "last": frames[-1]["date"] if frames else None,
        "available": bool(frames),
    }


def _num(v) -> float | None:
    try:
        x = float(str(v).replace(",", ""))
        if x != x:
            return None
        return x
    except (TypeError, ValueError):
        return None


def session_return(name: str, date: str, *, index_dir: Path | None = None) -> dict[str, Any]:
    """Close-to-close return ending on ``date``. Missing is UNKNOWN, not 0."""
    series = load_index(name, index_dir=index_dir)
    rows = series["rows"]
    by = {r["date"]: r for r in rows}
    if date not in by:
        return {
            "name": series["name"],
            "date": date,
            "ret": None,
            "status": "UNKNOWN",
            "reason": "session_not_in_local_files",
            "return_kind": series["return_kind"],
        }
    dates = [r["date"] for r in rows]
    i = dates.index(date)
    if i == 0:
        return {
            "name": series["name"],
            "date": date,
            "ret": None,
            "status": "UNKNOWN",
            "reason": "no_prior_close",
            "return_kind": series["return_kind"],
        }
    prev = rows[i - 1]["close"]
    cur = rows[i]["close"]
    if prev in (None, 0) or cur is None:
        return {
            "name": series["name"],
            "date": date,
            "ret": None,
            "status": "UNKNOWN",
            "reason": "bad_close",
            "return_kind": series["return_kind"],
        }
    return {
        "name": series["name"],
        "date": date,
        "ret": cur / prev - 1.0,
        "status": "OK",
        "return_kind": series["return_kind"],
    }


def catalog(index_dir: Path | None = None) -> dict[str, Any]:
    cov = file_coverage(index_dir)
    present = {}
    files = list_local_files(index_dir)
    names_in_file: set[str] = set()
    if files:
        import pandas as pd
        mid = files[len(files) // 2]
        try:
            df = pd.read_csv(mid, dtype=str)
            names_in_file = set(df["Index Name"].astype(str).str.strip())
        except Exception:
            names_in_file = set()
    for official, alias in PRICE_RETURN.items():
        present[alias] = {
            "official_name": official,
            "in_sample_file": official in names_in_file,
            "return_kind": "price_return",
        }
    tr = [n for n in names_in_file if any(h in n for h in TOTAL_RETURN_HINTS)]
    return {
        "coverage": cov,
        "price_return_indices": present,
        "total_return_names_seen": sorted(tr),
        "status": "RESEARCH_READY_WITH_LIMITATIONS" if cov["n_files"] else "UNUSABLE",
        "note": (
            "NSE daily close files are price-return unless the index name "
            "contains TR. Do not compare a dividend-excluding strategy to a "
            "total-return benchmark without saying so. Local files start "
            f"{cov.get('first')} — not a 2019 official archive."
        ),
        "network": False,
    }


def sector_index_context(as_of: str, lookback: int = 63) -> list[dict[str, Any]]:
    """Sector index return / rank / trend from official local series. Research only."""
    wanted = [
        "Nifty Bank", "Nifty IT", "Nifty Pharma", "Nifty FMCG",
        "Nifty Auto", "Nifty Metal", "Nifty Energy", "Nifty Realty",
    ]
    rows = []
    for name in wanted:
        series = load_index(name, as_of=as_of)
        closes = [r["close"] for r in series["rows"] if r["close"] is not None]
        if len(closes) < lookback + 1:
            rows.append({
                "name": name,
                "ret": None,
                "status": "UNKNOWN",
                "reason": "insufficient_history",
                "return_kind": "price_return",
            })
            continue
        ret = closes[-1] / closes[-lookback - 1] - 1.0
        sma = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
        trend = "UP" if sma is not None and closes[-1] > sma else "DOWN"
        rows.append({
            "name": name,
            "ret": ret,
            "last": closes[-1],
            "trend": trend,
            "status": "OK",
            "return_kind": "price_return",
            "as_of": as_of,
        })
    usable = [r for r in rows if r.get("ret") is not None]
    usable.sort(key=lambda r: r["ret"], reverse=True)
    for i, r in enumerate(usable, 1):
        r["sector_rank"] = i
        if usable:
            r["sector_rs"] = 100.0 * (len(usable) - i) / max(len(usable) - 1, 1)
    return rows
