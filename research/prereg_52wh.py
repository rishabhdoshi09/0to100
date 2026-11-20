"""Executable definition of the pre-registered thesis PR-2026-11-20-52WH (research, shadow-only).

``research/preregistered/pr_2026_11_20_52wh.json`` is the frozen authority.
This module only computes what the spec says. Its scores are point-in-time
(bars up to the decision session only), its outcomes use the exact official
10-session window, and it has no path to the paper book, the BUY gate or a
broker. The test suite pins the spec's SHA-256, so any edit after freezing
fails CI instead of silently changing the hypothesis.

Nothing here runs on a schedule. Wiring it into the shadow ledger is a
deployment, which needs the owner's approval.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SPEC_PATH = Path(__file__).resolve().parent / "preregistered" / "pr_2026_11_20_52wh.json"
#: SHA-256 of the spec bytes when it was frozen (2026-11-20). Never update it to "fix" a test.
FROZEN_SHA256 = "4a9133b2e2be630368b3d5b7d33bbd6c97badd401aec449ed1d9bd7cb1785597"


def spec_sha256(path: Path = SPEC_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def required_blocks(effect: float, sd: float, *, alpha: float = 0.05, power: float = 0.80) -> int:
    """Independent blocks a one-sided test at `alpha` needs to detect `effect` with `power` (normal approx)."""
    from statistics import NormalDist

    z = NormalDist().inv_cdf
    return int(math.ceil(((z(1 - alpha) + z(power)) * sd / effect) ** 2))


def decision_sessions(official_sessions: Iterable[str], spec: Mapping[str, Any]) -> list[str]:
    """Every horizon-th official session from the first eligible one (inclusive)."""
    first = str(spec["schedule"]["first_eligible_decision_session"])
    step = int(spec["schedule"]["horizon_sessions"])
    days = sorted(d for d in {str(x)[:10] for x in official_sessions} if d >= first)
    return days[::step]


def _is_member(member: Mapping[str, tuple[str, str | None]] | None, sym: str, day: str) -> bool:
    row = (member or {}).get(sym)
    if row is None:
        return False                                  # no membership row: not eligible (never assumed)
    listed, delisted = row
    return listed <= day and (delisted is None or delisted > day)


def score_session(frames: Mapping[str, Any], day: str, *, member: Mapping[str, tuple[str, str | None]] | None,
                  spec: Mapping[str, Any]) -> dict[str, float]:
    """{symbol: close(d) / 252-session high} for every eligible symbol, from bars up to `day` only."""
    import pandas as pd

    uni, sc = spec["universe"], spec["score"]
    lookback = int(sc["lookback_sessions"])
    cut = pd.Timestamp(day)
    out: dict[str, float] = {}
    for sym, df in frames.items():
        if df is None or not len(df) or not _is_member(member, sym, day):
            continue
        hist = df[df.index <= cut]
        if len(hist) < int(uni["min_history_sessions"]) or pd.Timestamp(hist.index[-1]).normalize() != cut:
            continue                                  # needs a bar ON the decision session
        close = float(hist["close"].iloc[-1])
        if not math.isfinite(close) or close < float(uni["min_price_rs"]):
            continue
        traded = (hist["close"] * hist["volume"]).iloc[-20:]
        if float(traded.median()) < float(uni["min_median_traded_value_rs_20d"]):
            continue
        high = float(hist["high"].iloc[-lookback:].max())
        if high > 0 and math.isfinite(high):
            out[sym] = close / high
    return out


def block_returns(frames: Mapping[str, Any], day: str, window: Sequence[str],
                  symbols: Iterable[str]) -> tuple[dict[str, float], int]:
    """Entry at the open of window[0], exit at the close of window[-1]; symbols missing any bar are excluded."""
    import pandas as pd

    want = [pd.Timestamp(d) for d in window]
    out: dict[str, float] = {}
    missing = 0
    for sym in symbols:
        df = frames.get(sym)
        if df is None:
            missing += 1
            continue
        bars = df[(df.index >= want[0]) & (df.index <= want[-1])]
        if [pd.Timestamp(x).normalize() for x in bars.index] != want:
            missing += 1
            continue
        entry, exit_ = float(bars["open"].iloc[0]), float(bars["close"].iloc[-1])
        if entry > 0 and math.isfinite(exit_):
            out[sym] = exit_ / entry - 1.0
        else:
            missing += 1
    return out, missing


def block_ic(scores: Mapping[str, float], returns: Mapping[str, float]) -> float | None:
    from scan.signal_backtest import spearman_ic

    common = sorted(set(scores) & set(returns))
    if len(common) < 3:
        return None
    return spearman_ic([scores[s] for s in common], [returns[s] for s in common])


def final_test(ics: Sequence[float], spec: Mapping[str, Any]) -> dict[str, Any]:
    """The one pre-registered analysis: one-sided t-test of mean IC > 0 at the frozen sample size."""
    from research.overlap_inference import _t_sf

    vals = [float(x) for x in ics if x is not None and math.isfinite(float(x))]
    need = int(spec["stopping"]["final_analysis_at_blocks"])
    if len(vals) < need:
        return {"status": "NOT_DUE", "blocks": len(vals), "required": need}
    vals = vals[:need]                                # never more than the frozen sample size
    n = len(vals)
    mean = sum(vals) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1))
    if sd == 0:
        return {"status": "DEGENERATE", "blocks": n}
    t = mean / (sd / math.sqrt(n))
    p = _t_sf(t, n - 1)
    alpha = float(spec["primary_outcome"]["alpha"])
    return {"status": "PASS" if p < alpha else "FAIL", "blocks": n, "mean_ic": round(mean, 4),
            "sd_ic": round(sd, 4), "t": round(t, 3), "p_one_sided": round(p, 5), "alpha": alpha}


def futility(ics: Sequence[float], spec: Mapping[str, Any]) -> bool:
    """Non-binding futility at the frozen look: mean IC <= 0 after exactly that many blocks."""
    look = int(spec["stopping"]["futility_look_at_blocks"])
    vals = [float(x) for x in ics if x is not None][:look]
    return len(vals) == look and sum(vals) / look <= 0


def membership_from_rows(rows: Iterable[Mapping[str, Any]] | None) -> dict[str, tuple[str, str | None]]:
    out: dict[str, tuple[str, str | None]] = {}
    for r in rows or []:
        try:
            out[str(r["symbol"]).upper()] = (str(r["listed"])[:10],
                                             str(r["delisted"])[:10] if r.get("delisted") else None)
        except Exception:
            continue
    return out


