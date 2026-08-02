"""
Correlation engine — kitne POSITIONS nahi, kitne asli BETS.

HAL + BEL + BHEL + L&T = chaar stocks, par ek hi defence/cap-goods macro
bet. The sector cap already catches same-sector stacking; this catches what
sectors miss — cross-sector names that MOVE together. It clusters open
positions by realised 60-day return correlation (bhavcopy history, no new
data source) and reports effective independent bets vs raw position count.

READ-ONLY: it informs the Brain and the book display. Position limits stay
where they are — this adds the lens, not a new gate (a gate can come later
once the lens proves itself).

Pure core (`clusters_from_corr`) takes a correlation lookup → fully testable
with synthetic series.
"""
from __future__ import annotations

from logger import get_logger

log = get_logger(__name__)

CLUSTER_RHO = 0.70            # ≥ this → same macro bet
_LOOKBACK = 60                # sessions of daily returns


def pairwise_corr(symbols: list[str], lookback: int = _LOOKBACK) -> dict:
    """{(a, b): rho} for every pair with enough overlapping history.
    Pairs with <30 overlapping sessions are omitted (no claim)."""
    try:
        import numpy as np
        from data.bhavcopy_store import get_ohlcv
    except Exception:
        return {}
    rets: dict[str, "np.ndarray"] = {}
    for s in symbols:
        try:
            df = get_ohlcv(s)
            if df is None or len(df) < 31:
                continue
            close = df["close"].astype(float).tail(lookback + 1)
            rets[s] = close.pct_change().dropna().to_numpy()
        except Exception:
            continue
    out: dict = {}
    syms = sorted(rets)
    for i, a in enumerate(syms):
        for b in syms[i + 1:]:
            ra, rb = rets[a], rets[b]
            n = min(len(ra), len(rb))
            if n < 30:
                continue
            try:
                import numpy as np
                rho = float(np.corrcoef(ra[-n:], rb[-n:])[0, 1])
                if rho == rho:                      # NaN guard
                    out[(a, b)] = round(rho, 3)
            except Exception:
                continue
    return out


def clusters_from_corr(symbols: list[str], corr: dict,
                       threshold: float = CLUSTER_RHO) -> list[list[str]]:
    """Union-find: merge symbols whose pairwise rho ≥ threshold. Returns
    clusters (sorted, singletons included) — len(clusters) = effective bets."""
    parent = {s: s for s in symbols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), rho in corr.items():
        if a in parent and b in parent and rho >= threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    groups: dict[str, list[str]] = {}
    for s in symbols:
        groups.setdefault(find(s), []).append(s)
    return sorted((sorted(g) for g in groups.values()),
                  key=lambda g: (-len(g), g))


def book_correlation_report() -> dict:
    """The live book through the correlation lens:
    {available, n_positions, n_bets, clusters: [[syms...]], biggest: [syms]|None}.
    Safe on empty book / missing history (falls back to positions=bets)."""
    try:
        from risk.position_manager import review_positions
        symbols = sorted({p.get("symbol") for p in (review_positions() or [])
                          if p.get("symbol")})
    except Exception:
        symbols = []
    if len(symbols) < 2:
        return {
            "available": True,
            "n_positions": len(symbols),
            "n_bets": len(symbols),
            "clusters": [[s] for s in symbols],
            "biggest": None,
            "message": (
                "Need at least two open positions to measure concentration."
                if len(symbols) < 2 else ""
            ),
        }
    corr = pairwise_corr(symbols)
    clusters = clusters_from_corr(symbols, corr)
    biggest = clusters[0] if clusters and len(clusters[0]) > 1 else None
    return {
        "available": True,
        "n_positions": len(symbols),
        "n_bets": len(clusters),
        "clusters": clusters,
        "biggest": biggest,
        "message": (
            f"{len(symbols)} positions behave like {len(clusters)} independent bet(s)."
        ),
    }
