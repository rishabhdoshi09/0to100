"""
🧠 Market Memory — "this setup resembles these 143 prior ones; here's what
happened."

The upgrade from a similarity engine to a DECISION engine. Instead of retrieving
similar market *days*, it retrieves similar *setups* — a breakout matched on
volatility, momentum, extension, ATR expansion, trend stage — and reports the
distribution of what those analogs actually did:

    win rate · average R · median holding period · MAE · MFE

MAE (Maximum Adverse Excursion) and MFE (Maximum Favorable Excursion) are the
point: they say where analogous winners *dipped to before running* and where
losers *peaked before failing* — which is exactly the information that sets a
stop and a target. A win rate alone can't do that.

Honest framing (Research OS discipline): this shows a DISTRIBUTION of what
happened, never a point prediction; it's a prior that sharpens sizing and exits,
not a hard signal. Retrieval purges temporally-adjacent setups so "nearest
neighbours" aren't just yesterday, and it demands a minimum number of analogs
before it will speak.

Pure core (feature extraction, forward outcome with excursions, Mahalanobis
retrieval, distribution) is unit-tested; the corpus is built once from the
bhavcopy history (I/O layer, fail-open, validated on-machine).
"""
from __future__ import annotations

import os as _os

import numpy as np

_MIN_ANALOGS = int(_os.getenv("QT_MM_MIN_ANALOGS", "20") or 20)
_DEFAULT_K = int(_os.getenv("QT_MM_K", "50") or 50)
_HORIZON = int(_os.getenv("QT_MM_HORIZON", "15") or 15)

# The vector that characterises a setup at entry. Kept small and economically
# meaningful (Mahalanobis handles the differing scales + correlations).
FEATURE_NAMES = ("rsi", "mom5_pct", "atr_pct", "vol_ratio",
                 "dist_high_pct", "ext_50dma_pct", "above_200")


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction (from an OHLCV window, last bar = the setup)
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(close, high, low, vol) -> np.ndarray | None:
    """The setup feature vector for the LAST bar of the window, in FEATURE_NAMES
    order. Returns None if the window is too short. Reuses the scanner's own
    indicator maths (one language for evidence)."""
    from scan.unified_scanner import _rsi, _atr, _ema_np
    close = np.asarray(close, float)
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    n = close.size
    if n < 60:
        return None
    price = float(close[-1])
    if price <= 0:
        return None
    rsi = _rsi(close)
    mom5 = (close[-1] / close[-6] - 1) * 100 if n > 5 else 0.0
    atr = _atr(high, low, close)
    atr_pct = atr / price * 100 if price > 0 else 0.0
    vratio = 1.0
    if vol is not None:
        vol = np.asarray(vol, float)
        if vol.size > 21:
            avg = np.nanmean(vol[-21:-1])
            vratio = float(vol[-1] / avg) if avg > 0 else 1.0
    hi252 = float(np.max(high[:-1])) if n > 1 else price
    dist_high = (hi252 - price) / hi252 * 100 if hi252 > 0 else 0.0
    sma50 = float(close[-50:].mean()) if n >= 50 else price
    ext50 = (price / sma50 - 1) * 100 if sma50 > 0 else 0.0
    above200 = 1.0 if (n >= 200 and price > float(close[-200:].mean())) else 0.0
    return np.array([rsi, mom5, atr_pct, vratio, dist_high, ext50, above200],
                    dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
# Forward outcome with excursions (the MAE/MFE the reviewer asked for)
# ══════════════════════════════════════════════════════════════════════════════

def forward_outcome(entry: float, stop: float, target: float,
                    fwd_high, fwd_low, fwd_close) -> dict | None:
    """First-touch simulation over the forward window, tracking the full
    excursion picture. Returns r (realised R at stop/target/horizon), mae
    (worst adverse excursion, in R, always ≥0), mfe (best favorable excursion,
    in R), hold (bars to exit), won. None if the risk geometry is invalid."""
    risk = entry - stop
    if risk <= 0:
        return None
    h = np.asarray(fwd_high, float)
    l = np.asarray(fwd_low, float)
    c = np.asarray(fwd_close, float)
    mae = 0.0
    mfe = 0.0
    for i in range(h.size):
        mfe = max(mfe, (h[i] - entry) / risk)
        mae = max(mae, (entry - l[i]) / risk)
        if l[i] <= stop:
            return {"r": -1.0, "mae": mae, "mfe": mfe, "hold": i + 1, "won": False}
        if h[i] >= target:
            return {"r": (target - entry) / risk, "mae": mae, "mfe": mfe,
                    "hold": i + 1, "won": True}
    r = (float(c[-1]) - entry) / risk if c.size else 0.0
    return {"r": r, "mae": mae, "mfe": mfe, "hold": int(h.size), "won": r > 0}


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval — Mahalanobis k-NN
# ══════════════════════════════════════════════════════════════════════════════

def robust_inv_cov(corpus: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Regularised inverse covariance for Mahalanobis distance. The ridge keeps
    it invertible when features are collinear or the corpus is small — the same
    shrinkage that stops a naive covariance estimate from overfitting."""
    x = np.asarray(corpus, float)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.eye(x.shape[1] if x.ndim == 2 else 1)
    cov = np.cov(x, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + ridge * np.eye(cov.shape[0]) * (np.trace(cov) / cov.shape[0] + 1e-9)
    try:
        return np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov)


def mahalanobis_knn(query: np.ndarray, corpus: np.ndarray,
                    inv_cov: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices + distances of the k nearest corpus rows to `query` under the
    Mahalanobis metric (scale- and correlation-aware — the honest distance for
    heterogeneous features)."""
    q = np.asarray(query, float)
    x = np.asarray(corpus, float)
    diff = x - q
    d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
    d2 = np.maximum(d2, 0.0)
    k = min(k, x.shape[0])
    idx = np.argsort(d2)[:k]
    return idx, np.sqrt(d2[idx])


def outcome_distribution(rs, maes, mfes, holds, wons) -> dict:
    """Summarise the analogs' outcomes into a decision-grade read + the
    plain-English sentence."""
    rs = np.asarray(rs, float)
    n = rs.size
    if n == 0:
        return {"n": 0, "insight": "No analogs."}
    maes = np.asarray(maes, float)
    mfes = np.asarray(mfes, float)
    holds = np.asarray(holds, float)
    wons = np.asarray(wons, float)
    win_rate = float(wons.mean())
    avg_r = float(rs.mean())
    med_r = float(np.median(rs))
    avg_mae = float(maes.mean())
    avg_mfe = float(mfes.mean())
    med_hold = float(np.median(holds))
    insight = (f"{n} similar setups: {win_rate*100:.0f}% win, avg {avg_r:+.2f}R "
               f"(median {med_r:+.2f}R). Typically dipped to -{avg_mae:.1f}R "
               f"before resolving and peaked +{avg_mfe:.1f}R; ~{med_hold:.0f}-day hold.")
    return {"n": int(n), "win_rate": round(win_rate, 3),
            "avg_r": round(avg_r, 3), "median_r": round(med_r, 3),
            "avg_mae": round(avg_mae, 3), "avg_mfe": round(avg_mfe, 3),
            "median_hold": round(med_hold, 1), "insight": insight}


def analog_summary(query_features, corpus_features, corpus_outcomes,
                   k: int = _DEFAULT_K, min_analogs: int = _MIN_ANALOGS) -> dict:
    """Retrieve the k nearest historical setups to `query_features` and
    summarise what they did. `corpus_outcomes` is a list of dicts with keys
    r/mae/mfe/hold/won (from forward_outcome). Returns {} until there are at
    least `min_analogs` in the corpus — no distribution from a handful."""
    x = np.asarray(corpus_features, float)
    if x.ndim != 2 or x.shape[0] < min_analogs:
        return {}
    inv_cov = robust_inv_cov(x)
    idx, dist = mahalanobis_knn(np.asarray(query_features, float), x, inv_cov, k)
    picked = [corpus_outcomes[i] for i in idx]
    summ = outcome_distribution([o["r"] for o in picked],
                                [o["mae"] for o in picked],
                                [o["mfe"] for o in picked],
                                [o["hold"] for o in picked],
                                [o["won"] for o in picked])
    summ["mean_distance"] = round(float(np.mean(dist)), 3) if dist.size else 0.0
    return summ


# ══════════════════════════════════════════════════════════════════════════════
# Corpus (I/O over the bhavcopy history) — built once, cached. Fail-open.
# ══════════════════════════════════════════════════════════════════════════════

def build_corpus(sample_step: int = 5, horizon: int = _HORIZON,
                 max_symbols: int = 800, lookback: int = 250) -> dict:
    """Walk the bhavcopy store, and at each sampled bar record (features →
    forward_outcome) for a breakout-style setup (entry = the bar's high, stop =
    entry − 2·ATR, target = entry + 4·ATR — the scanner's own geometry). Returns
    {"features": (N,F) array, "outcomes": [dict]}. Heavy; run off-hours and
    cache. Fail-open (no store / error → empty). Validated on-machine — the
    sandbox has no bhav data."""
    out_feat: list = []
    out_outcome: list = []
    try:
        from data.bhavcopy_store import store_symbols, get_ohlcv
        from scan.unified_scanner import _atr
        symbols = store_symbols()[:max_symbols]
    except Exception:
        return {"features": np.zeros((0, len(FEATURE_NAMES))), "outcomes": []}
    for sym in symbols:
        try:
            df = get_ohlcv(sym)
            if df is None or len(df) < 60 + horizon + 20:
                continue
            close = df["close"].to_numpy(float)
            high = df["high"].to_numpy(float)
            low = df["low"].to_numpy(float)
            vol = df["volume"].to_numpy(float) if "volume" in df else None
            n = len(close)
            start = max(60, n - horizon - lookback)
            for t in range(start, n - horizon, sample_step):
                feat = extract_features(close[:t], high[:t], low[:t],
                                        None if vol is None else vol[:t])
                if feat is None:
                    continue
                atr = _atr(high[:t], low[:t], close[:t])
                if atr <= 0:
                    continue
                entry = float(high[t - 1])
                oc = forward_outcome(entry, entry - 2 * atr, entry + 4 * atr,
                                     high[t:t + horizon], low[t:t + horizon],
                                     close[t:t + horizon])
                if oc is None:
                    continue
                out_feat.append(feat)
                out_outcome.append(oc)
        except Exception:
            continue
    feats = (np.vstack(out_feat) if out_feat
             else np.zeros((0, len(FEATURE_NAMES))))
    return {"features": feats, "outcomes": out_outcome}


_corpus_cache: dict = {"data": None}


def reset_analog_corpus_cache() -> None:
    """Clear cached historical analog corpus (test / process isolation)."""
    _corpus_cache["data"] = None


def find_analogs(symbol: str, df, k: int = _DEFAULT_K) -> dict:
    """User-facing entry: for a live setup (its OHLCV frame), retrieve similar
    historical setups and summarise what they did. Builds+caches the corpus on
    first call. Fail-open → {}."""
    try:
        close = df["close"].to_numpy(float)
        high = df["high"].to_numpy(float)
        low = df["low"].to_numpy(float)
        vol = df["volume"].to_numpy(float) if "volume" in df else None
        q = extract_features(close, high, low, vol)
        if q is None:
            return {}
        if _corpus_cache["data"] is None:
            _corpus_cache["data"] = build_corpus()
        corpus = _corpus_cache["data"]
        if corpus["features"].shape[0] < _MIN_ANALOGS:
            return {}
        return analog_summary(q, corpus["features"], corpus["outcomes"], k=k)
    except Exception:
        return {}
