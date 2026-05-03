"""Corporate actions – dividends, splits.  yfinance primary."""
from __future__ import annotations

from typing import Any


from sq_ai.backend.cache import cached


@cached("actions", ttl_seconds=86400)
def get_actions(symbol: str) -> dict[str, Any]:
    out = {"symbol": symbol, "dividends": [], "splits": [],
           "buybacks": [], "bonuses": [], "source": "fallback"}
    try:
        import yfinance as yf  # noqa: WPS433
        t = yf.Ticker(symbol)
        out["source"] = "yfinance"
        divs = t.dividends
        if divs is not None and not divs.empty:
            out["dividends"] = [
                {"date": str(d.date() if hasattr(d, "date") else d),
                 "amount": round(float(v), 4)}
                for d, v in divs.tail(20).items()
            ]
        splits = t.splits
        if splits is not None and not splits.empty:
            out["splits"] = [
                {"date": str(d.date() if hasattr(d, "date") else d),
                 "ratio": float(v)}
                for d, v in splits.tail(10).items()
            ]
    except Exception:
        pass
    return out


__all__ = ["get_actions"]
